use ndarray::prelude::*;
use enum_as_inner::EnumAsInner;
use crate::scc::mixer::BroydenMixer;

/// A `Property` is a piece of data that can be associated with an `Molecule` or
/// `ElectronicData`. The idea of this enum is taken from Guillaume Fraux's (@Luthaf) Chemfiles
/// library. The original implementation can be found on:
/// [Github](https://github.com/chemfiles/chemfiles.rs/blob/master/src/property.rs)
/// The functionality of the `Property` enum is expanded by the use of the `EnumAsInner` macro.
/// This allows to get direct access to the inner values of the enum without doing
/// case matching. As an example the inner fields can be accessed by using the methods `into_$name()`
/// or `as_$name()`. e.g: (see [Documentation of enum-as-inner](https://docs.rs/enum-as-inner/0.3.3/enum_as_inner/)
/// for details).
/// ## Basic example for Bool and Array1
///
///  ```rust
///  let flag: bool = true;
///  let prop1: Property = Property::from(flag);
///  assert_eq!(prop1.as_bool().unwrap(), &true);
///  assert_eq!(prop1.into_bool().unwrap(), true);
///
///  let vector: Array1<f64> = Array1::zeros([4]);
///  let prop2: Property = Property::from(vector);
///  assert_eq!(prop2.as_array1.unwrap(), &Array1::zeros([4]));
///  assert_eq!(prop2.into_array1.unwrap(), Array::zeros([4]));
///  ```
///
///
#[derive(Debug, Clone, EnumAsInner)]
pub enum Property {
    /// Boolean property
    Bool(bool),
    /// Floating point property
    Double(f64),
    /// String property
    String(String),
    /// Vector property of u8 type
    VecU8(Vec<u8>),
    /// Vector property of usize type
    VecUsize(Vec<usize>),
    /// Vector property of f64 type
    VecF64(Vec<f64>),
    /// Arraybase<f64, Ix1> property
    Array1(Array1<f64>),
    /// Arraybase<f64, Ix2> property
    Array2(Array2<f64>),
    /// Arraybase<f64, Ix3> property
    Array3(Array3<f64>),
    /// Arraybase<bool, Ix2> property
    Array2Bool(Array2<bool>),
    /// SCC Mixer property
    Mixer(BroydenMixer),
}

impl Default for Property {
    fn default() -> Self {
       Property::Bool(false)
    }
}

impl From<bool> for Property {
    fn from(value: bool) -> Self {
        Property::Bool(value)
    }
}

impl From<f64> for Property {
    fn from(value: f64) -> Self {
        Property::Double(value)
    }
}

impl From<String> for Property {
    fn from(value: String) -> Self {
        Property::String(value)
    }
}

impl From<Vec<u8>> for Property {
    fn from(value: Vec<u8>) -> Self {Property::VecU8(value)}
}
impl From<Vec<usize>> for Property {
    fn from(value: Vec<usize>) -> Self {Property::VecUsize(value)}
}

impl From<Vec<f64>> for Property {
    fn from(value: Vec<f64>) -> Self {Property::VecF64(value)}
}

impl<'a> From<&'_ str> for Property {
    fn from(value: &'_ str) -> Self {
        Property::String(value.into())
    }
}

impl From<Array1<f64>> for Property {
    fn from(value: Array1<f64>) -> Self {
        Property::Array1(value)
    }
}

impl From<ArrayView1<'_, f64>> for Property {
    fn from(value: ArrayView1<'_, f64>) -> Self {
        Property::Array1(value.to_owned())
    }
}

impl From<Array2<f64>> for Property {
    fn from(value: Array2<f64>) -> Self {
        Property::Array2(value)
    }
}

impl From<ArrayView2<'_, f64>> for Property {
    fn from(value: ArrayView2<'_, f64>) -> Self {
        Property::Array2(value.to_owned())
    }
}

impl From<Array3<f64>> for Property {
    fn from(value: Array3<f64>) -> Self {
        Property::Array3(value)
    }
}

impl From<ArrayView3<'_, f64>> for Property {
    fn from(value: ArrayView3<'_, f64>) -> Self {
        Property::Array3(value.to_owned())
    }
}


impl From<Array2<bool>> for Property {
    fn from(value: Array2<bool>) -> Self {
        Property::Array2Bool(value)
    }
}

impl From<ArrayView2<'_, bool>> for Property {
    fn from(value: ArrayView2<'_, bool>) -> Self {
        Property::Array2Bool(value.to_owned())
    }
}

impl From<BroydenMixer> for Property {
    fn from(value: BroydenMixer) -> Self {
        Property::Mixer(value)
    }
}
