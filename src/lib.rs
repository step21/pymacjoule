#![allow(non_upper_case_globals)]
//#![allow(dead_code)]
//#![allow(unused_imports)]
//#![allow(unused_variables)]
#![allow(unused_assignments)]

use pyo3::prelude::*;
// atm unused
//use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyRuntimeError;

use core_foundation::dictionary::CFDictionaryRef;
use serde::Serialize;

mod sources;
use crate::sources::{
  cfio_get_residencies, cfio_watts, libc_ram, libc_swap, IOHIDSensors, IOReport, SocInfo, SMC,
};

type WithError<T> = Result<T, Box<dyn std::error::Error>>;

// const CPU_FREQ_DICE_SUBG: &str = "CPU Complex Performance States";
const CPU_FREQ_CORE_SUBG: &str = "CPU Core Performance States";
const GPU_FREQ_DICE_SUBG: &str = "GPU Performance States";




/// A Python module implemented in Rust.
#[pymodule]
fn pymacjoule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Sampler>()?;
    m.add_class::<TempMetrics>()?;
    m.add_class::<MemMetrics>()?;
    m.add_class::<Metrics>()?;
    Ok(())
}

// The python module implemented in rust
/*#[pymodule]
fn rs_pyjoule(m: &Bound<'_, &PyModule>) -> PyResult<()> {
    m.add_class::<Sampler>()?;
    m.add_class::<TempMetrics>()?;
    m.add_class::<MemMetrics>()?;
    m.add_class::<Metrics>()?;
    // Add any other classes or functions you want to expose to Python
    Ok(())
} */

// MARK: Structs
#[pyclass]
#[derive(Debug, Default, Serialize, Clone)]
pub struct TempMetrics {
  #[pyo3(get)]
  pub cpu_temp_avg: f32, // Celsius
  #[pyo3(get)]
  pub gpu_temp_avg: f32, // Celsius
}

// Implement conversion to Python object
/*impl IntoPy<PyObject> for TempMetrics {
  fn into_py(self, py: Python) -> PyObject {
      let dict = pyo3::types::PyDict::new(py);
      dict.set_item("cpu_temp_avg", self.cpu_temp_avg).unwrap();
      dict.set_item("gpu_temp_avg", self.gpu_temp_avg).unwrap();
      dict.into()
  }
}*/

#[pyclass]
#[derive(Debug, Default, Serialize, Clone)]
pub struct MemMetrics {
  #[pyo3(get)]
  pub ram_total: u64,  // bytes
  #[pyo3(get)]
  pub ram_usage: u64,  // bytes
  #[pyo3(get)]
  pub swap_total: u64, // bytes
  #[pyo3(get)]
  pub swap_usage: u64, // bytes
}

#[pymethods]
impl MemMetrics {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "MemMetrics(ram_usage={}/{}, swap_usage={}/{})",
            self.ram_usage, self.ram_total,
            self.swap_usage, self.swap_total
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyclass]
#[derive(Debug, Default, Serialize)]
pub struct Metrics {
  #[pyo3(get)]
  pub temp: TempMetrics,
  #[pyo3(get)]
  pub memory: MemMetrics,
  #[pyo3(get)]
  pub ecpu_usage: (u32, f32), // freq, percent_from_max
  #[pyo3(get)]
  pub pcpu_usage: (u32, f32), // freq, percent_from_max
  #[pyo3(get)]
  pub gpu_usage: (u32, f32),  // freq, percent_from_max
  #[pyo3(get)]
  pub cpu_power: f32,         // Watts
  #[pyo3(get)]
  pub gpu_power: f32,         // Watts
  #[pyo3(get)]
  pub ane_power: f32,         // Watts
  #[pyo3(get)]
  pub all_power: f32,         // Watts
  #[pyo3(get)]
  pub sys_power: f32,         // Watts
  #[pyo3(get)]
  pub ram_power: f32,         // Watts
  #[pyo3(get)]
  pub gpu_ram_power: f32,     // Watts
}

#[pymethods]
impl Metrics {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "Metrics(Temperatur={:?}, memory={:?}, ecpu_usage={:?}, pcpu_usage={:?}, gpu_usage={:?}, cpu_power={}, gpu_power={}, ane_power={}, all_power={}, sys_power={}, ram_power={}, gpu_ram_power={}",
            self.temp, self.memory, self.ecpu_usage, self.pcpu_usage, self.gpu_usage, self.cpu_power, self.gpu_power, self.ane_power, self.all_power, self.sys_power, self.ram_power, self.gpu_ram_power,
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

// MARK: Helpers

pub fn zero_div<T: core::ops::Div<Output = T> + Default + PartialEq>(a: T, b: T) -> T {
  let zero: T = Default::default();
  return if b == zero { zero } else { a / b };
}

fn calc_freq(item: CFDictionaryRef, freqs: &Vec<u32>) -> (u32, f32) {
  let items = cfio_get_residencies(item); // (ns, freq)
  let (len1, len2) = (items.len(), freqs.len());
  assert!(len1 > len2, "cacl_freq invalid data: {} vs {}", len1, len2); // todo?

  // IDLE / DOWN for CPU; OFF for GPU; DOWN only on M2?/M3 Max Chips
  let offset = items.iter().position(|x| x.0 != "IDLE" && x.0 != "DOWN" && x.0 != "OFF").unwrap();

  let usage = items.iter().map(|x| x.1 as f64).skip(offset).sum::<f64>();
  let total = items.iter().map(|x| x.1 as f64).sum::<f64>();
  let count = freqs.len();

  let mut avg_freq = 0f64;
  for i in 0..count {
    let percent = zero_div(items[i + offset].1 as _, usage);
    avg_freq += percent * freqs[i] as f64;
  }

  let usage_ratio = zero_div(usage, total);
  let min_freq = freqs.first().unwrap().clone() as f64;
  let max_freq = freqs.last().unwrap().clone() as f64;
  let from_max = (avg_freq.max(min_freq) * usage_ratio) / max_freq;

  (avg_freq as u32, from_max as f32)
}

fn calc_freq_final(items: &Vec<(u32, f32)>, freqs: &Vec<u32>) -> (u32, f32) {
  let avg_freq = zero_div(items.iter().map(|x| x.0 as f32).sum(), items.len() as f32);
  let avg_perc = zero_div(items.iter().map(|x| x.1 as f32).sum(), items.len() as f32);
  let min_freq = freqs.first().unwrap().clone() as f32;

  (avg_freq.max(min_freq) as u32, avg_perc)
}

fn init_smc() -> WithError<(SMC, Vec<String>, Vec<String>)> {
  let mut smc = SMC::new()?;
  const FLOAT_TYPE: u32 = 1718383648; // FourCC: "flt "

  let mut cpu_sensors = Vec::new();
  let mut gpu_sensors = Vec::new();

  let names = smc.read_all_keys().unwrap_or(vec![]);
  for name in &names {
    let key = match smc.read_key_info(&name) {
      Ok(key) => key,
      Err(_) => continue,
    };

    if key.data_size != 4 || key.data_type != FLOAT_TYPE {
      continue;
    }

    let _ = match smc.read_val(&name) {
      Ok(val) => val,
      Err(_) => continue,
    };

    // Unfortunately, it is not known which keys are responsible for what.
    // Basically in the code that can be found publicly "Tp" is used for CPU and "Tg" for GPU.

    match name {
      // "Tp" – performance cores, "Te" – efficiency cores
      name if name.starts_with("Tp") || name.starts_with("Te") => cpu_sensors.push(name.clone()),
      name if name.starts_with("Tg") => gpu_sensors.push(name.clone()),
      _ => (),
    }
  }

  // println!("{} {}", cpu_sensors.len(), gpu_sensors.len());
  Ok((smc, cpu_sensors, gpu_sensors))
}

// MARK: Sampler

#[pyclass(unsendable)]
#[derive(Debug)]
pub struct Sampler {
  #[pyo3(get)]
  soc: SocInfo,
  ior: IOReport,
  hid: IOHIDSensors,
  smc: SMC,
  #[pyo3(get)]
  smc_cpu_keys: Vec<String>,
  #[pyo3(get)]
  smc_gpu_keys: Vec<String>,
}

#[pymethods]
impl Sampler {
  #[new]
  pub fn new() -> PyResult<Self> {
    let channels = vec![
      ("Energy Model", None), // cpu/gpu/ane power
      // ("CPU Stats", Some(CPU_FREQ_DICE_SUBG)), // cpu freq by cluster
      ("CPU Stats", Some(CPU_FREQ_CORE_SUBG)), // cpu freq per core
      ("GPU Stats", Some(GPU_FREQ_DICE_SUBG)), // gpu freq
    ];

    let soc = SocInfo::new()
      .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ior = IOReport::new(channels)
      .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let hid = IOHIDSensors::new()
      .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (smc, smc_cpu_keys, smc_gpu_keys) = init_smc()
      .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(Sampler { soc, ior, hid, smc, smc_cpu_keys, smc_gpu_keys })
  }

  fn get_temp_smc(&mut self) -> PyResult<TempMetrics> {
    let mut cpu_metrics = Vec::new();
    for sensor in &self.smc_cpu_keys {
      let val = self.smc.read_val(sensor)?;
      let val = f32::from_le_bytes(val.data[0..4].try_into().unwrap());
      cpu_metrics.push(val);
    }

    let mut gpu_metrics = Vec::new();
    for sensor in &self.smc_gpu_keys {
      let val = self.smc.read_val(sensor)?;
      let val = f32::from_le_bytes(val.data[0..4].try_into().unwrap());
      gpu_metrics.push(val);
    }

    let cpu_temp_avg = zero_div(cpu_metrics.iter().sum::<f32>(), cpu_metrics.len() as f32);
    let gpu_temp_avg = zero_div(gpu_metrics.iter().sum::<f32>(), gpu_metrics.len() as f32);

    Ok(TempMetrics { cpu_temp_avg, gpu_temp_avg })
  }

  fn get_temp_hid(&mut self) -> PyResult<TempMetrics> {
    let metrics = self.hid.get_metrics();

    let mut cpu_values = Vec::new();
    let mut gpu_values = Vec::new();

    for (name, value) in &metrics {
      if name.starts_with("pACC MTR Temp Sensor") || name.starts_with("eACC MTR Temp Sensor") {
        // println!("{}: {}", name, value);
        cpu_values.push(*value);
        continue;
      }

      if name.starts_with("GPU MTR Temp Sensor") {
        // println!("{}: {}", name, value);
        gpu_values.push(*value);
        continue;
      }
    }

    let cpu_temp_avg = zero_div(cpu_values.iter().sum(), cpu_values.len() as f32);
    let gpu_temp_avg = zero_div(gpu_values.iter().sum(), gpu_values.len() as f32);

    Ok(TempMetrics { cpu_temp_avg, gpu_temp_avg })
  }

  fn get_temp(&mut self) -> PyResult<TempMetrics> {
    // HID for M1, SMC for M2/M3
    // UPD: Looks like HID/SMC related to OS version, not to the chip (SMC available from macOS 14)
    match self.smc_cpu_keys.len() > 0 {
      true => self.get_temp_smc(),
      false => self.get_temp_hid(),
    }
  }

  fn get_mem(&mut self) -> PyResult<MemMetrics> {
    let (ram_usage, ram_total) = libc_ram()?;
    let (swap_usage, swap_total) = libc_swap()?;
    Ok(MemMetrics { ram_total, ram_usage, swap_total, swap_usage })
  }

  fn get_sys_power(&mut self) -> PyResult<f32> {
    let val = self.smc.read_val("PSTR")?;
    let val = f32::from_le_bytes(val.data.clone().try_into().unwrap());
    Ok(val)
  }

  pub fn get_metrics(&mut self, duration: u32) -> PyResult<Metrics> {
    let measures: usize = 4;
    let mut results: Vec<Metrics> = Vec::with_capacity(measures);

    // do several samples to smooth metrics
    // see: https://github.com/vladkens/macmon/issues/10
    for (sample, dt) in self.ior.get_samples(duration as u64, measures) {
      let mut ecpu_usages = Vec::new();
      let mut pcpu_usages = Vec::new();
      let mut rs = Metrics::default();

      for x in sample {
        if x.group == "CPU Stats" && x.subgroup == CPU_FREQ_CORE_SUBG {
          if x.channel.contains("ECPU") {
            ecpu_usages.push(calc_freq(x.item, &self.soc.ecpu_freqs));
            continue;
          }

          if x.channel.contains("PCPU") {
            pcpu_usages.push(calc_freq(x.item, &self.soc.pcpu_freqs));
            continue;
          }
        }

        if x.group == "GPU Stats" && x.subgroup == GPU_FREQ_DICE_SUBG {
          match x.channel.as_str() {
            "GPUPH" => rs.gpu_usage = calc_freq(x.item, &self.soc.gpu_freqs[1..].to_vec()),
            _ => {}
          }
        }

        if x.group == "Energy Model" {
          match x.channel.as_str() {
            "GPU Energy" => rs.gpu_power += cfio_watts(x.item, &x.unit, dt)?,
            // "CPU Energy" for Basic / Max, "DIE_{}_CPU Energy" for Ultra
            c if c.ends_with("CPU Energy") => rs.cpu_power += cfio_watts(x.item, &x.unit, dt)?,
            // same pattern next keys: "ANE" for Basic, "ANE0" for Max, "ANE0_{}" for Ultra
            c if c.starts_with("ANE") => rs.ane_power += cfio_watts(x.item, &x.unit, dt)?,
            c if c.starts_with("DRAM") => rs.ram_power += cfio_watts(x.item, &x.unit, dt)?,
            c if c.starts_with("GPU SRAM") => rs.gpu_ram_power += cfio_watts(x.item, &x.unit, dt)?,
            _ => {}
          }
        }
      }

      rs.ecpu_usage = calc_freq_final(&ecpu_usages, &self.soc.ecpu_freqs);
      rs.pcpu_usage = calc_freq_final(&pcpu_usages, &self.soc.pcpu_freqs);
      results.push(rs);
    }

    let mut rs = Metrics::default();
    rs.ecpu_usage.0 = zero_div(results.iter().map(|x| x.ecpu_usage.0).sum(), measures as _);
    rs.ecpu_usage.1 = zero_div(results.iter().map(|x| x.ecpu_usage.1).sum(), measures as _);
    rs.pcpu_usage.0 = zero_div(results.iter().map(|x| x.pcpu_usage.0).sum(), measures as _);
    rs.pcpu_usage.1 = zero_div(results.iter().map(|x| x.pcpu_usage.1).sum(), measures as _);
    rs.gpu_usage.0 = zero_div(results.iter().map(|x| x.gpu_usage.0).sum(), measures as _);
    rs.gpu_usage.1 = zero_div(results.iter().map(|x| x.gpu_usage.1).sum(), measures as _);
    rs.cpu_power = zero_div(results.iter().map(|x| x.cpu_power).sum(), measures as _);
    rs.gpu_power = zero_div(results.iter().map(|x| x.gpu_power).sum(), measures as _);
    rs.ane_power = zero_div(results.iter().map(|x| x.ane_power).sum(), measures as _);
    rs.ram_power = zero_div(results.iter().map(|x| x.ram_power).sum(), measures as _);
    rs.gpu_ram_power = zero_div(results.iter().map(|x| x.gpu_ram_power).sum(), measures as _);
    rs.all_power = rs.cpu_power + rs.gpu_power + rs.ane_power;

    rs.memory = self.get_mem()?;
    rs.temp = self.get_temp()?;

    rs.sys_power = match self.get_sys_power() {
      Ok(val) => val.max(rs.all_power),
      Err(_) => 0.0,
    };

    Ok(rs)
  }
}
