use ndarray::prelude::*;

/// A `Property` is a piece of data that can be associated with an `Molecule` or a
/// `ElectronicData`. The idea of this enum is taken from Guillaume Fraux's (@Luthaf) Chemfiles
/// library. The original implementation can be found on:
/// [Github](https://github.com/chemfiles/chemfiles.rs/blob/master/src/property.rs)
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Property {
    /// Boolean property
    Bool(bool),
    /// Floating point property
    Double(f64),
    /// String property
    String(String),
    /// Arraybase<f64, Ix1> property
    Array1D(Array1<f64>),
    /// Arraybase<f64, Ix1> property
    Array2D(Array2<f64>),
    /// Arraybase<f64, Ix1> property
    Array3D(Array3<f64>),
    /// Arraybase<bool, Ix2> property
    Array2DBool(Array2<bool>)
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

impl<'a> From<&'a str> for Property {
    fn from(value: &'a str) -> Self {
        Property::String(value.into())
    }
}

impl From<Array1<f64>> for Property {
    fn from(value: Array1<f64>) -> Self {
        Property::Array1D(value)
    }
}

impl From<ArrayView1<f64>> for Property {
    fn from(value: ArrayView1<f64>) -> Self {
        Property::Array1D(value.to_owned())
    }
}

impl From<Array2<f64>> for Property {
    fn from(value: Array2<f64>) -> Self {
        Property::Array2D(value)
    }
}

impl From<ArrayView2<f64>> for Property {
    fn from(value: ArrayView2<f64>) -> Self {
        Property::Array2D(value.to_owned())
    }
}

impl From<Array3<f64>> for Property {
    fn from(value: Array3<f64>) -> Self {
        Property::Array3D(value)
    }
}

impl From<ArrayView3<f64>> for Property {
    fn from(value: ArrayView3<f64>) -> Self {
        Property::Array3D(value.to_owned())
    }
}


impl From<Array2<bool>> for Property {
    fn from(value: Array2<bool>) -> Self {
        Property::Array2DBool(value)
    }
}

impl From<ArrayView2<bool>> for Property {
    fn from(value: ArrayView2<bool>) -> Self {
        Property::Array2DBool(value.to_owned())
    }
}

impl Property {
    pub(crate) fn as_ref(&self) ->  {
        match *self {
            Property::Bool(value) => RawProperty::bool(value),
            Property::Double(value) => RawProperty::double(value),
            Property::String(ref value) => RawProperty::string(value),
            Property::Vector3D(value) => RawProperty::vector3d(value),
        }
    }