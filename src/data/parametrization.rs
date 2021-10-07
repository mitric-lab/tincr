use crate::data::ParamData;

impl<'a> ParamData<'a> {
    pub fn new() -> Self {
        Self {
            slako: None,
            vrep: None,
        }
    }

    /// Clear all data without any exceptions.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}