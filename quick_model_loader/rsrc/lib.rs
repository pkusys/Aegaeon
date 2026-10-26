use memmap2::MmapOptions;
use pyo3::exceptions::{PyBufferError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use safetensors::tensor::TensorInfo;
use safetensors::SafeTensors;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;

#[derive(Debug, Serialize, Default)]
struct SafeTensorsMeta {
    file_path: String,
    size: usize,
    offset: usize,
    tensor_info_map: HashMap<String, TensorInfo>,
}

impl SafeTensorsMeta {
    fn new(file_path: String) -> PyResult<Self> {
        let file = File::open(&file_path)?;
        let buffer = unsafe { MmapOptions::new().map(&file)? };
        let size = buffer.len();
        let mut offset = 8;
        let mut tensor_info_map = HashMap::new();

        let (n, metadata) = SafeTensors::read_metadata(&buffer).map_err(|_| {
            PyBufferError::new_err("Fail to read the metadata from the given buffer")
        })?;

        offset += n;

        for (name, tenser_info) in metadata.tensors() {
            tensor_info_map.insert(name, tenser_info.clone());
        }
        Ok(SafeTensorsMeta {
            file_path,
            size,
            offset,
            tensor_info_map,
        })
    }
}

#[pyfunction]
fn read_safetensors_meta_as_json(file_path: String) -> PyResult<String> {
    let safetensors_mata = SafeTensorsMeta::new(file_path)?;
    let meta_string = serde_json::to_string_pretty(&safetensors_mata)
        .map_err(|_| PyRuntimeError::new_err("Fail to serialize the safetensors meta"))?;
    Ok(meta_string)
}

#[pymodule]
fn _rlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_safetensors_meta_as_json, m)?)?;
    Ok(())
}
