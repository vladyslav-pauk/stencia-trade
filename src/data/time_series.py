import numpy as np


def generate_time_series(
        signal_type="periodic",
        length=1,
        critical_time=None,
        snr=0,
        amplitude=1,
        frequency=1,
        amplitude_ratio=1,
        frequency_ratio=np.pi,
        exponential_factor=-1,
        nonlinearity=0.5,
        alpha=0.5,
        seed=None
):
    """
    :param signal_type: str, Type of signal to generate. Options are:
        - "periodic": Simple periodic signal.
        - "quasi-periodic": Sum of two periodic signals with different frequencies.
        - "oscillatory": Oscillatory signal with a critical time.
        - "log-periodic-power": LPPLS signal with a critical time.
        - "geometric_random_walk": Geometric random walk.
        - "financial_crash": Financial signal with a stochastic component and an LPPLS crash.
        - "random": Random noise.
    :param length: int, Total length of the signal in time units.
    :param critical_time: float, critical transition length in time units.
    :param snr: float, Signal-to-noise ratio in decibels (dB).
    :param amplitude: float, expected value of the log-signal at the critical time.
    :param frequency:
    :param amplitude_ratio:
    :param frequency_ratio:
    :param exponential_factor: float, B < 0 corresponds to a super-exponential signal growth as the time approaches critical,
        while B > 0 corresponds to a super-exponential decay.
    :param nonlinearity: float, m = 0 corresponds to a strong super-linear trend of log-signal,
        while m = 1 corresponds to a nearly linear trend.
    :param alpha: float, Weighting factor between stochastic and LPPLS components (0 to 1).
    :param seed:
    :return: t, signal
    """
    if seed is not None:
        np.random.seed(seed)

    time = np.linspace(0, 1, length)
    periods_per_time_unit = 2 * np.pi * frequency

    def generate_noise(length, snr, signal):
        # noise = amplitude * snr * np.random.normal(size=length)
        signal_power = np.sum(signal ** 2) / length
        noise_power = signal_power / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)
        noise = noise_std * np.random.normal(size=length)
        return noise

    if signal_type == "periodic":
        signal = amplitude / 2 * (1 + np.sin(periods_per_time_unit * time))
        signal += generate_noise(length, snr, signal)
        return time, signal

    elif signal_type == "quasi-periodic":
        signal = (amplitude / 2 * (1 + np.sin(periods_per_time_unit * time))
                  + amplitude * amplitude_ratio * np.sin(periods_per_time_unit * frequency_ratio * time))
        signal += generate_noise(length, snr, signal)
        return time, signal

    elif signal_type == "oscillatory":
        critical_time = 1
        time = np.append(time[time < critical_time - 1e-6], critical_time - 1e-6)
        signal = amplitude / 2 * (1 + np.sin(periods_per_time_unit * frequency_ratio * np.log(critical_time - time)))
        signal += generate_noise(length, snr, signal)[:len(time)]
        return time, signal

    elif signal_type == "log-periodic-power":
        critical_time = 1
        if critical_time is None:
            raise ValueError("tc must be provided for LPPLS signals.")
        time = np.append(time[time < critical_time - 1e-6], critical_time - 1e-6)

        trend = amplitude * np.exp(exponential_factor * (critical_time - time) ** nonlinearity)
        oscillations = amplitude * amplitude_ratio * (critical_time - time) ** nonlinearity * np.cos(periods_per_time_unit * frequency_ratio**2 * np.log(critical_time - time))
        signal = trend + oscillations
        signal += generate_noise(length, snr, signal)[:len(time)]

        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        return time, signal

    elif signal_type == "geometric_brownian_motion":
        sigma = amplitude_ratio
        return generate_gbm_data(length, frequency_ratio, nonlinearity, sigma, amplitude)

    elif signal_type == "jump_diffusion":
        jump_intensity = amplitude_ratio
        jump_mean = alpha
        jump_std = amplitude * amplitude_ratio
        noise_power = amplitude / (10 ** (snr / 10))
        sigma = np.sqrt(noise_power)
        return generate_jump_diffusion_data(length, frequency_ratio, nonlinearity, sigma, amplitude, jump_intensity, jump_mean, jump_std)

    elif signal_type == "geometric_random_walk":
        signal = [amplitude]
        noise_power = amplitude / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)
        for i in range(length - 1):
            signal.append(signal[-1] * (1 + np.random.normal(0, noise_std)))
        return time, np.array(signal)

    elif signal_type == "random_walk_critical":
        if critical_time is None:
            raise ValueError("tc must be provided for LPPLS signals.")
        critical_time *= length

        signal = [amplitude]
        for _ in range(1, length):
            signal.append(signal[-1] * (1 + np.random.normal(0, 0.01)))
        stochastic_signal = np.array(signal)

        trend = np.zeros(length)
        oscillations = np.zeros(length)
        for i in range(length):
            if i >= 0:
                time_to_crash = critical_time - i
                if time_to_crash > 0:
                    trend[i] = amplitude * np.exp(exponential_factor * (time_to_crash / critical_time) ** nonlinearity)
                    oscillations[i] = (amplitude * amplitude_ratio * (i / critical_time) ** nonlinearity * (time_to_crash
                                       / critical_time) ** nonlinearity * np.cos(periods_per_time_unit * frequency_ratio * np.log(time_to_crash / critical_time) + frequency_ratio))
                else:
                    trend[i] = - (amplitude * (np.exp(- (i - critical_time) / length) - np.exp(- (length - critical_time) / length))
                                / (1 - np.exp((length - critical_time) / length)))
                    oscillations[i] = 0

        lppls_signal = trend + oscillations

        combined_signal = (1 - alpha) * stochastic_signal + alpha * lppls_signal

        combined_signal += generate_noise(length, snr, combined_signal)[:len(time)]

        combined_signal = np.nan_to_num(combined_signal, nan=0.0, posinf=0.0, neginf=0.0)
        return time, combined_signal

    elif signal_type == "random":
        return time, generate_noise(length, 0, amplitude)
    else:
        raise ValueError("Unknown signal type.")


def generate_jump_diffusion_data(timesteps, dt, mu, sigma, S0, jump_intensity, jump_mean, jump_std):
    """
    Generates time-series data for a Jump-Diffusion process.

    Args:
        timesteps (int): Number of timesteps to simulate.
        dt (float): Time increment.
        mu (float): Drift coefficient.
        sigma (float): Volatility coefficient.
        S0 (float): Initial price.
        jump_intensity (float): Poisson intensity (average number of jumps per unit time).
        jump_mean (float): Mean of the jump size (log-space).
        jump_std (float): Standard deviation of the jump size (log-space).

    Returns:
        t (np.ndarray): Time array.
        S (np.ndarray): Simulated price process.
    """
    t = np.arange(0, timesteps * dt, dt)
    S = np.zeros_like(t)
    S[0] = S0

    for i in range(1, len(t)):
        dW = np.sqrt(dt) * np.random.randn()
        jump_occurred = np.random.poisson(jump_intensity * dt) > 0
        jump_size = np.random.normal(jump_mean, jump_std) if jump_occurred else 0

        S[i] = S[i - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * dW + jump_size
        )

    return t, S


def generate_gbm_data(timesteps, dt, mu, sigma, S0):
    """Generates time-series data for Geometric Brownian Motion."""
    t = np.arange(0, timesteps * dt, dt)
    S = np.zeros_like(t)

    # Initial condition
    S[0] = S0

    # GBM simulation
    for i in range(1, len(t)):
        dW = np.sqrt(dt) * np.random.randn()
        S[i] = S[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)

    return t, S

# def generate_nsde_data(timesteps, dt, noise_std, spike_length, spike_amplitude_range, spike_interval_range):
#     """Generates time-series data for a noisy nonlinear oscillator."""
#     t = np.arange(0, timesteps * dt, dt)
#     x = np.zeros_like(t)
#     v = np.zeros_like(t)
#
#     # Initial conditions
#     x[0] = config['initial_conditions']['x0']
#     v[0] = config['initial_conditions']['v0']
#
#     # Coefficients for nonlinear oscillator
#     def damping_coefficient(x):
#         return config['coefficients']['damping_base'] + config['coefficients']['damping_scale'] * np.sin(x)
#
#     def stiffness_coefficient(x):
#         return config['coefficients']['stiffness_base'] + config['coefficients']['stiffness_scale'] * x ** 2
#
#     def nonlinear_term(x):
#         return config['coefficients']['nonlinear_scale'] * np.tanh(x)
#
#     def external_force(t):
#         """Decaying saw-like external force with irregular timing for spikes."""
#         force = np.zeros_like(t)
#         noise = np.random.randn(len(t)) * config['force']['noise_scale']
#
#         # Generate random start indices for spikes
#         spike_starts = np.cumsum(np.random.randint(*spike_interval_range, size=(len(t) // 500,)))
#         spike_starts = spike_starts[spike_starts < len(t)]
#
#         for start in spike_starts:
#             spike_amp = np.random.uniform(*spike_amplitude_range)
#             spike = spike_amp * np.linspace(1, -0.5, spike_length)
#             if (start // 50) % 2 == 0:
#                 spike = -spike
#             force[start:start + len(spike)] += spike[:len(force) - start]
#
#         return force + noise
#
#     # Time evolution
#     F_t = external_force(t)
#     for i in range(1, len(t)):
#         a_x = (
#             -damping_coefficient(x[i - 1]) * v[i - 1]
#             - stiffness_coefficient(x[i - 1]) * x[i - 1]
#             + nonlinear_term(x[i - 1])
#             + F_t[i - 1]
#         )
#         v[i] = v[i - 1] + a_x * dt + noise_std * np.sqrt(dt) * np.random.randn()
#         x[i] = x[i - 1] + v[i - 1] * dt + noise_std * np.sqrt(dt) * np.random.randn()
#
#     return t, x, F_t


# todo: add other typical time series generation functions: step, GW dynamics, GW filters, finance patters, etc.
