from dataclasses import dataclass, field
import numpy as np

generator = np.random.default_rng(1)

@dataclass
class TemperatureData:
    num_readings: int = field(init=False)
    mean_temp: float = field(init=False)
    std_dev_temp: float = field(init=False)
    data: np.ndarray = field(init=False)
    mean: float = field(init=False)
    median: float = field(init=False)
    std: float = field(init=False)
    fifth_percentile: float = field(init=False)
    ninetyfifth_percentile: float = field(init=False)

    def __post_init__(self):
        self.num_readings = generator.integers(1_000_000, 10_000_001)
        self.mean_temp = generator.uniform(40, 50)
        self.std_dev_temp = generator.uniform(10, 15)
        self.data = generator.normal(
            loc=self.mean_temp,
            scale=self.std_dev_temp,
            size=self.num_readings
        )
        self.mean = np.mean(self.data)
        self.median = np.median(self.data)
        self.std = np.std(self.data)
        self.fifth_percentile = np.percentile(self.data, 5)
        self.ninetyfifth_percentile = np.percentile(self.data, 95)


    def __repr__(self):
        return (
            '--- Temperature Statistics ---\n'
            f'Mean (Average): {self.mean:.2f} °C\n'
            f'Median (Middle): {self.median:.2f} °C\n'
            f'Std. Deviation (Spread): {self.std:.2f} °C\n'
            f'90% Normal Range: {self.fifth_percentile:.2f} °C'
            f'to {self.ninetyfifth_percentile:.2f} °C'
        )


if __name__ == '__main__':
    x = TemperatureData()
    print(x)
