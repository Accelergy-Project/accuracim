"""Noise profiles. To add a profile, subclass `Noise` in your own file (e.g. in `workloads/`).

A profile is a callable `noise(x, low=..., high=...)`. It returns an error tensor shaped like `x`.
`low` and `high` give the range of the perturbed value (bit slice, ADC output).
With them, `StateIndependentNoise` can use "a fixed fraction of full scale".
For several profiles on one tensor, use a list. Their errors add (`apply_noise`),
and `estimate` treats each profile as a separate source.

A profile has one strength and, optionally, fixed shape parameters:
- The strength is the value that `noise * k` scales. It alone sets `get_scaling_factor`.
- Shape parameters (`shape()`) do not change. A different shape is a different source.
`check_profile` verifies this. A profile without `get_scaling_factor` runs only in precise mode.

Configs name a profile by class (`type: ProportionalNoise`). Classes in this file are found
automatically. For a class in your own file, use a "module:Class" path.
`repr` is the cache key. A custom `__repr__` must be deterministic and unique per class.

`normal_like` and `uniform_like` use the thread-local seed from `set_noise_seed`.
A calibration run and the sweep that reuses its cached errors thus get the same random draw.
"""

import hashlib
import math
import threading

import torch

_NOISE_RNG = threading.local()


def _get_noise_generator(device: torch.device) -> torch.Generator | None:
    """Return the seeded per-device generator, or None if no seed is set."""
    seed = getattr(_NOISE_RNG, "seed", None)
    if seed is None:
        return None
    generators = getattr(_NOISE_RNG, "generators", None)
    if generators is None:
        generators = {}
        _NOISE_RNG.generators = generators
    key = str(device)
    if key not in generators:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        generators[key] = gen
    return generators[key]


def _randn_like_compat(tensor: torch.Tensor, generator: torch.Generator | None):
    """`randn_like` with `generator`, with fallbacks for older torch (last resort: unseeded)."""
    if generator is None:
        return torch.randn_like(tensor)
    try:
        return torch.randn_like(tensor, generator=generator)
    except TypeError:
        try:
            return torch.randn(
                tensor.size(), generator=generator, dtype=tensor.dtype, device=tensor.device
            )
        except TypeError:
            return torch.randn_like(tensor)


def set_noise_seed(seed: int | None):
    """Set the thread-local noise seed (None clears it); return the old state for `restore_noise_seed`."""
    prev_seed = getattr(_NOISE_RNG, "seed", None)
    prev_generators = getattr(_NOISE_RNG, "generators", None)
    if seed is None:
        if hasattr(_NOISE_RNG, "seed"):
            delattr(_NOISE_RNG, "seed")
        if hasattr(_NOISE_RNG, "generators"):
            delattr(_NOISE_RNG, "generators")
        return (prev_seed, prev_generators)
    _NOISE_RNG.seed = int(seed)
    _NOISE_RNG.generators = {}
    return (prev_seed, prev_generators)


def restore_noise_seed(state):
    prev_seed, prev_generators = state
    if prev_seed is None:
        if hasattr(_NOISE_RNG, "seed"):
            delattr(_NOISE_RNG, "seed")
        if hasattr(_NOISE_RNG, "generators"):
            delattr(_NOISE_RNG, "generators")
        return
    _NOISE_RNG.seed = prev_seed
    _NOISE_RNG.generators = prev_generators if prev_generators is not None else {}


def normal_like(tensor: torch.Tensor, std: float | torch.Tensor = 1.0):
    """Sample N(0, std^2) shaped like `tensor`, seeded if `set_noise_seed` is active."""
    gen = _get_noise_generator(tensor.device)
    return _randn_like_compat(tensor, gen) * std


def uniform_like(tensor: torch.Tensor):
    """Sample U[0, 1) shaped like `tensor`, seeded if `set_noise_seed` is active."""
    gen = _get_noise_generator(tensor.device)
    if gen is None:
        return torch.rand_like(tensor)
    return torch.rand(tensor.size(), generator=gen, dtype=tensor.dtype, device=tensor.device)


def apply_noise(noise, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Error of one profile, or the summed errors of a list of profiles."""
    if isinstance(noise, (list, tuple)):
        error = torch.zeros_like(x)
        for profile in noise:
            error = error + profile(x, **kwargs)
        return error
    return noise(x, **kwargs)


def noise_label(noise: "Noise") -> str:
    """Source name of a profile: its class, plus its shape if it has one."""
    shape = noise.shape()
    return type(noise).__name__ + (f"({', '.join(shape)})" if shape else "")


def check_profile(noise: "Noise", k: float = 2.0) -> None:
    """Raise ValueError unless `noise` follows the one-parameter contract (see module doc)."""
    name = noise_label(noise)
    try:
        scaled = noise * k
        base_factor, scaled_factor = noise.get_scaling_factor(), scaled.get_scaling_factor()
    except (NotImplementedError, AttributeError) as exc:
        raise ValueError(f"{name}: needs `__mul__` and `get_scaling_factor` for estimate") from exc

    problems = []
    if scaled.shape() != noise.shape():
        problems.append(f"`noise * {k}` changes the shape {noise.shape()} -> {scaled.shape()}")
    if not math.isclose(scaled_factor, k**2 * base_factor, rel_tol=1e-6, abs_tol=1e-30):
        problems.append(
            f"scaling factor of `noise * {k}` is {scaled_factor:.6g}, "
            f"expected {k ** 2:g} x {base_factor:.6g}"
        )

    x = torch.linspace(-1.0, 1.0, 257)
    state = set_noise_seed(12345)
    try:
        error = noise(x, low=-1.0, high=1.0)
        set_noise_seed(12345)
        scaled_error = scaled(x, low=-1.0, high=1.0)
    except RuntimeError:  # e.g. a per-element std tensor that fits only its own layer
        error = scaled_error = None
    finally:
        restore_noise_seed(state)
    if error is not None and not torch.allclose(scaled_error, k * error, rtol=1e-5, atol=1e-8):
        problems.append(f"with the same random draw, `noise * {k}` does not give {k} x the error")

    if problems:
        raise ValueError(f"{name} is not a one-parameter profile: " + "; ".join(problems))


class Noise:
    """Base profile; subclasses define `__call__`, `__mul__`, and `get_scaling_factor`."""

    #: False: resampled every forward pass. True: applied once when cells are
    #: programmed (see `ConductanceDrift`).
    stationary = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # `repr` keys the cache (`params.get_cache_dir`), so classes must not share one:
        # a subclass without its own `__repr__` gets the class-named default, not its parent's.
        if "__repr__" not in cls.__dict__:
            cls.__repr__ = Noise.__repr__

    def __repr__(self):
        """`ClassName(attr=value, ...)`: deterministic, as it keys the cache."""
        attrs = [
            f"{k}={v!r}"
            for k, v in sorted(vars(self).items())
            if not (k == "func" and v is None) and not (k == "padding" and v == 0)
        ]
        return f"{type(self).__name__}({', '.join(attrs)})"

    def __init__(self, func=None):
        self.func = func
        self.padding = 0

    def __call__(self, x, **kwargs):
        return self.func(x, **kwargs)

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, scalar):
        raise NotImplementedError

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __getitem__(self, idx_range):
        return self

    def shape(self) -> tuple:
        """Fixed parameters that select the error profile, as `name=value` strings.

        Profiles with different shapes are different noise sources (see module doc).
        """
        return ()

    def add_padding(self, padding: int = 0):
        self.padding = padding
        for _, attr in vars(self).items():
            if isinstance(attr, Noise):
                attr.add_padding(padding)

    def copy(self):
        copied = self.__class__.__new__(self.__class__)
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(copied, attr, value.clone())
            elif isinstance(value, Noise):
                setattr(copied, attr, value.copy())
            else:
                setattr(copied, attr, value)
        return copied

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for key in self.__dict__:
            val1 = getattr(self, key)
            val2 = getattr(other, key, None)
            if isinstance(val1, torch.Tensor):
                if not isinstance(val2, torch.Tensor) or not torch.equal(val1, val2):
                    return False
            elif isinstance(val1, Noise):
                if not val1 == val2:
                    return False
            else:
                if val1 != val2:
                    return False
        for key in other.__dict__:
            if key not in self.__dict__:
                return False
        return True

    def __hash__(self):
        def tensor_hash(t):
            return hash(t.cpu().numpy().tobytes())

        attrs = []
        for key in sorted(self.__dict__):
            val = getattr(self, key)
            if isinstance(val, torch.Tensor):
                attrs.append((key, tensor_hash(val)))
            elif isinstance(val, Noise):
                attrs.append((key, hash(val)))
            else:
                attrs.append((key, hash(val)))
        return hash((self.__class__, tuple(attrs)))


class ProportionalNoise(Noise):
    """Signal-proportional: std = alpha * x."""

    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = abs(alpha)

    def get_scaling_factor(self):
        return (self.alpha) ** 2

    def __call__(self, x, **kwargs):
        return normal_like(x, x * self.alpha)

    def __repr__(self):
        return f"alpha={self.alpha}"

    def __mul__(self, scalar):
        return ProportionalNoise(self.alpha * scalar)


class StateIndependentNoise(Noise):
    """Fixed fraction of full scale: std = alpha * (high - low)."""

    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = abs(alpha)

    def get_scaling_factor(self):
        return (self.alpha) ** 2

    def __call__(self, x, low=0, high=0, **kwargs):
        return normal_like(x, self.alpha * abs(high - low))

    def __repr__(self):
        return f"beta={self.alpha}"

    def __mul__(self, scalar):
        return StateIndependentNoise(self.alpha * scalar)


class GaussianNoise(Noise):
    """Fixed std, scalar or per-element tensor."""

    def __init__(self, std=0):
        super().__init__()
        self.std = abs(std)

    def get_scaling_factor(self):
        return (
            torch.mean(self.std**2).item()
            if isinstance(self.std, torch.Tensor)
            else (self.std) ** 2
        )

    def __call__(self, x, **kwargs):
        std = self.std
        if isinstance(self.std, torch.Tensor):
            std = std.to(x.device)
            if std.shape != x.shape:
                std = std.expand_as(x)
        return normal_like(x, std)

    def __getitem__(self, idx_range):
        if not isinstance(idx_range, (tuple, list)):
            idx_range = (idx_range,)
        if isinstance(self.std, torch.Tensor):
            return GaussianNoise(self.std[idx_range])
        return GaussianNoise(self.std)

    def _pattern(self) -> str:
        """Short hash of the per-element std pattern (std / mean std): the profile's shape."""
        std = self.std.detach().double().cpu()
        normalized = std / std.mean().clamp_min(1e-300)
        return hashlib.sha256(normalized.numpy().tobytes()).hexdigest()[:12]

    def shape(self):
        # Tensor std: the pattern is fixed, only its overall level is the strength
        if isinstance(self.std, torch.Tensor):
            return (f"pattern={self._pattern()}", f"size={tuple(self.std.shape)}")
        return ()

    def __repr__(self):
        if isinstance(self.std, torch.Tensor):
            # The pattern hash keeps different tensors with the same mean apart (cache key)
            return f"std={torch.mean(self.std)}, shape={self.std.shape}, pattern={self._pattern()}"
        return str(self.std)

    def __mul__(self, scalar):
        return GaussianNoise(self.std * scalar)


class ConductanceDrift(Noise):
    """Programmed cells losing conductance: `W -> W(1-alpha)`.

    Stationary and deterministic: applied once when weights are programmed
    (`quantize.slice_weights`), never redrawn. `__call__` returns the error
    `-alpha*x`, matching the other profiles' contract. Being signal-correlated,
    `sources` folds it into the ideal baseline rather than treating it as an
    additive source (see `sources.STATIONARY_NOISE_KEYS`).
    """

    stationary = True

    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = abs(alpha)

    def get_scaling_factor(self):
        return (self.alpha) ** 2

    def __call__(self, x, **kwargs):
        return -self.alpha * x

    def __repr__(self):
        return f"drift={self.alpha}"

    def __mul__(self, scalar):
        return ConductanceDrift(self.alpha * scalar)


class PowerLawDrift(ConductanceDrift):
    """PCM conductance drift: G(t) = G(t0) * (t / t0)^(-nu), so `W -> W(1 - alpha)`.

    alpha = 1 - (t / t0)^(-nu); stationary like `ConductanceDrift` (applied once when
    programmed, folded into the baseline). One drift coefficient `nu` for every cell;
    per-cell variation is not modeled.

    Ielmini et al., "Recovery and drift dynamics of resistance and threshold voltages in
    phase-change memories," IEEE TED 54(2), 2007; Joshi et al., Nat. Commun. 11, 2473 (2020).
    """

    def __init__(self, t=1.0, nu=0.05, t0=1.0):
        if t < t0 or t0 <= 0 or nu < 0:
            raise ValueError(
                f"PowerLawDrift needs t >= t0 > 0 and nu >= 0 (got t={t}, t0={t0}, nu={nu})"
            )
        super().__init__(1.0 - (t / t0) ** (-nu))
        self.t, self.nu, self.t0 = t, nu, t0

    def __mul__(self, scalar):
        # Scales the decay itself (the strength), as `ConductanceDrift` does
        return ConductanceDrift(self.alpha * scalar)


class ShotNoise(Noise):
    """Shot noise of a read current: variance proportional to the signal, std = alpha * sqrt(|x| * FS).

    FS = high - low, so at full scale std = alpha * FS (as `StateIndependentNoise` there).
    Schottky (1918); current shot noise sigma_I^2 = 2 q I df.
    """

    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = abs(alpha)

    def get_scaling_factor(self):
        return self.alpha**2

    def __call__(self, x, low=0, high=0, **kwargs):
        return normal_like(x, self.alpha * torch.sqrt(x.abs() * abs(high - low)))

    def __mul__(self, scalar):
        return ShotNoise(self.alpha * scalar)


class TelegraphNoise(Noise):
    """Random telegraph noise: a trapped charge lowers conductance by a fraction alpha.

    error = -alpha * x * b, b ~ Bernoulli(p) redrawn on every read. Strength: alpha.
    Shape: p (trap occupancy). Scaling factor alpha^2 * p (mean squared relative error).
    Ielmini, Nardi, Cagli, "Resistance-dependent amplitude of random telegraph-signal
    noise in resistive switching memories," Appl. Phys. Lett. 96, 053503 (2010).
    """

    def __init__(self, alpha=0, p=0.5):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"TelegraphNoise needs 0 <= p <= 1 (got {p})")
        self.alpha = abs(alpha)
        self.p = p

    def get_scaling_factor(self):
        return self.alpha**2 * self.p

    def shape(self):
        return (f"p={self.p}",)

    def __call__(self, x, **kwargs):
        trapped = (uniform_like(x) < self.p).to(x.dtype)
        return -self.alpha * x * trapped

    def __mul__(self, scalar):
        return TelegraphNoise(self.alpha * scalar, self.p)


class ADCGain(Noise):
    """ADC gain error: error = gamma * x (the same transfer error on every conversion).

    For `adc_physical_noise`. Kester, Data Conversion Handbook (Analog Devices, 2005).
    """

    def __init__(self, gamma=0):
        super().__init__()
        self.gamma = gamma

    def get_scaling_factor(self):
        return self.gamma**2

    def __call__(self, x, **kwargs):
        return self.gamma * x

    def __mul__(self, scalar):
        return ADCGain(self.gamma * scalar)


class ADCOffset(Noise):
    """ADC offset error: error = offset * FS on every conversion (FS = high - low).

    For `adc_physical_noise`. Kester, Data Conversion Handbook (Analog Devices, 2005).
    """

    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def get_scaling_factor(self):
        return self.offset**2

    def __call__(self, x, low=0, high=0, **kwargs):
        return torch.ones_like(x) * (self.offset * abs(high - low))

    def __mul__(self, scalar):
        return ADCOffset(self.offset * scalar)


class ADCNonlinearity(Noise):
    """ADC integral nonlinearity (INL), a cubic bow with peak |error| = a * FS.

    error = a * FS * (3 sqrt(3) / 2) * (u^3 - u), u = 2 (x - low) / FS - 1, clamped to
    [-1, 1] (no error outside the range; clipping is modeled separately).
    For `adc_physical_noise`. Kester, Data Conversion Handbook (Analog Devices, 2005).
    """

    _PEAK = 3 * math.sqrt(3) / 2  # 1 / max |u^3 - u| on [-1, 1]

    def __init__(self, a=0):
        super().__init__()
        self.a = a

    def get_scaling_factor(self):
        return self.a**2

    def __call__(self, x, low=0, high=0, **kwargs):
        full_scale = abs(high - low)
        if not full_scale:
            return torch.zeros_like(x)
        u = (2 * (x - low) / full_scale - 1).clamp(-1, 1)
        return (self.a * full_scale * self._PEAK) * (u**3 - u)

    def __mul__(self, scalar):
        return ADCNonlinearity(self.a * scalar)
