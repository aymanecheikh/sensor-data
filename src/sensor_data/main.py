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

    def __repr__(self, data_name: tuple):
        return (
            f'--- {data_name[0]} Statistics ---\n'
            f'Mean (Average): {self.mean:.2f} {data_name[1]}\n'
            f'Median (Middle): {self.median:.2f} {data_name[1]}\n'
            f'Std. Deviation (Spread): {self.std:.2f} {data_name[1]}\n'
            f'90% Normal Range: {self.fifth_percentile:.2f} {data_name[1]} '
            f'to {self.ninetyfifth_percentile:.2f} {data_name[1]}\n'
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
        self.status_codes = generator.choice(
            a=np.arange(4),
            size=self.num_readings,
            p=[0.85, 0.10, 0.03, 0.02]
        )
        self.severity_threshold = self.stats.mean + (3 * self.stats.std)
        self.critical_status_mask = (self.status_codes == 2)
        self.high_temp_outlier_mask = (self.data > self.severity_threshold)
        self.critical_anomaly_mask = (
            self.critical_status_mask & self.high_temp_outlier_mask
        )
        self.extracted_anomalies = self.data[self.critical_anomaly_mask]
        self.anomaly_count = self.critical_anomaly_mask.sum()
        self.valid_data_mask = (self.status_codes != 3)
        self.valid_median = np.median(self.data[self.valid_data_mask])
        self.cleaned_temperature_data = np.where(
            self.status_codes == 3,
            self.valid_median,
            self.data
        )
        self.imputed_count = (self.status_codes == 3).sum()

    def __repr__(self):
        return (
            f'{self.stats.__repr__(('Temperature', '°C'))}'
            f'Median of All Valid Readings: {self.valid_median:.2f} °C\n'
            f'Severe Outlier Threshold: {self.severity_threshold:.2f} °C\n'
            f'Critical Status Readings: {self.critical_status_mask.sum()}\n'
            f'High-temp outliers: {self.high_temp_outlier_mask.sum()}\n'
            f'Total Critical Anomalies: {self.anomaly_count}\n'
            f'Sample Temperatures: {self.extracted_anomalies[:5]}\n'
            f'Total Faulty readings imputed: {self.imputed_count}'
        )

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
        return self.stats.__repr__(('Pressure', 'kPa'))


if __name__ == '__main__':
    t = TemperatureData()
    p = PressureData()
