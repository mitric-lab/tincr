use crate::param::Element;
use std::path::Path;
use std::fs;
use std::collections::HashMap;

#[derive(Clone)]
pub struct SkfHandler {
    pub el_a: Element,
    pub el_b: Element,
    pub data_string: String,
}

impl SkfHandler {
    pub fn new(el_a: Element, el_b: Element, path_prefix: &str) -> SkfHandler {
        let element_1: &str = el_a.symbol();
        let element_2: &str = el_b.symbol();
        let filename: String = format!("{}/{}-{}.skf", path_prefix, element_1, element_2);
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");

        SkfHandler {
            el_a: el_a,
            el_b: el_b,
            data_string: data,
        }
    }
}

/// Converts a line into a list of column values respecting the  format conventions used
/// in the Slater-Koster files (.skf). Note: In these files, zero columns are not written!
/// e.g. 4*0.0 has to be replaced by four columns with zeros "0.0 0.0 0.0 0.0".
pub fn process_slako_line(line: &str) -> Vec<f64> {
    let line: String = line.replace(",", " ");
    let new_line: Vec<&str> = line.split(" ").collect();
    let mut float_vec: Vec<f64> = Vec::new();
    for string in new_line {
        if string.contains("*") {
            let temp: Vec<&str> = string.split("*").collect();
            let count: usize = temp[0].trim().parse::<usize>().unwrap();
            let value: f64 = temp[1].trim().parse::<f64>().unwrap();
            for it in (0..count) {
                float_vec.push(value);
            }
        } else {
            if string.len() > 0 && string.contains("\t") == false {
                // println!("string {:?}",string);
                let value: f64 = string.trim().parse::<f64>().unwrap();
                float_vec.push(value);
            }
        }
    }
    return float_vec;
}

pub fn get_tau_2_index(tuple: (u8, i32, u8, i32)) -> u8 {
    let v1: u8 = tuple.0;
    let v2: i32 = tuple.1;
    let v3: u8 = tuple.2;
    let v4: i32 = tuple.3;
    let value: u8 = match (v1, v2, v3, v4) {
        (0, 0, 0, 0) => 0,
        (0, 0, 1, 0) => 2,
        (0, 0, 2, 0) => 3,
        (1, 0, 0, 0) => 4,
        (1, -1, 1, -1) => 5,
        (1, 0, 1, 0) => 6,
        (1, 1, 1, 1) => 5,
        (1, -1, 2, -1) => 7,
        (1, 0, 2, 0) => 8,
        (1, 1, 2, 1) => 7,
        (2, 0, 0, 0) => 9,
        (2, -1, 1, -1) => 10,
        (2, 0, 1, 0) => 11,
        (2, 1, 1, 1) => 10,
        (2, -2, 2, -2) => 12,
        (2, -1, 2, -1) => 13,
        (2, 0, 2, 0) => 14,
        (2, 1, 2, 1) => 13,
        (2, 2, 2, 2) => 12,
        _ => panic!("false combination for tau_2_index!"),
    };
    return value;
}

pub fn get_index_to_symbol() -> HashMap<u8, String> {
    let mut index_to_symbol: HashMap<u8, String> = HashMap::new();
    index_to_symbol.insert(0, String::from("ss_sigma"));
    index_to_symbol.insert(2, String::from("ss_sigma"));
    index_to_symbol.insert(3, String::from("sp_sigma"));
    index_to_symbol.insert(4, String::from("sd_sigma"));
    index_to_symbol.insert(5, String::from("ps_sigma"));
    index_to_symbol.insert(6, String::from("pp_pi"));
    index_to_symbol.insert(7, String::from("pp_sigma"));
    index_to_symbol.insert(8, String::from("pd_pi"));
    index_to_symbol.insert(9, String::from("pd_sigma"));
    index_to_symbol.insert(10, String::from("ds_sigma"));
    index_to_symbol.insert(11, String::from("dp_pi"));
    index_to_symbol.insert(12, String::from("dp_sigma"));
    index_to_symbol.insert(13, String::from("dd_delta"));
    index_to_symbol.insert(14, String::from("dd_pi"));
    index_to_symbol
}
