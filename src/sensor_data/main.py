from dataclasses import dataclass, field
import numpy as np
generator = np.random.default_rng(1)


@dataclass
class DataDescriptor:
    data: np.ndarray

    def __post_init__(self):
        self.mean = np.mean(self.data)
        self.median = np.median(self.data)
        self.std = np.std(self.data)
        self.fifth_percentile = np.percentile(self.data, 5)
        self.ninetyfifth_percentile = np.percentile(self.data, 95)

    def __repr__(self, data_name):
        return (
            f'--- {data_name} Statistics ---\n'
            f'Mean (Average): {self.mean:.2f} °C\n'
            f'Median (Middle): {self.median:.2f} °C\n'
            f'Std. Deviation (Spread): {self.std:.2f} °C\n'
            f'90% Normal Range: {self.fifth_percentile:.2f} °C '
            f'to {self.ninetyfifth_percentile:.2f} °C\n'
        )


class GenData:
    num_readings: int = generator.integers(1_000_000, 10_000_001)

@dataclass
class TemperatureData(GenData):
    mean_temp: float = field(default=generator.uniform(40, 50), init=False)
    std_dev_temp: float = field(default=generator.uniform(10, 15), init=False)

    def __post_init__(self):
        self.data = generator.normal(
            loc=self.mean_temp,
            scale=self.std_dev_temp,
            size=self.num_readings
        )
        self.stats = DataDescriptor(self.data)

    def __repr__(self):
        return self.stats.__repr__('Temperature')

@dataclass
class PressureData(GenData):
    def __post_init__(self):
        self.data = generator.uniform(
            low=100,
            high=500,
            size=self.num_readings
        )
        self.stats = DataDescriptor(self.data)

    def __repr__(self):
        return self.stats.__repr__('Pressure')


if __name__ == '__main__':
    t = TemperatureData()
    p = PressureData()
