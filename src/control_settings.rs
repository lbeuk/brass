use std::time::Duration;

/// Type of EventReciever to use for the application.
pub enum EventRecieverType {
    Null,
    Threaded,
}

/// Settings for configuring the behavior of an ApplicationController instance.
pub struct ApplicationControlSettings {
    /// Whether to have application automatically exit when Crtl + C is pressed.
    pub ctrl_c_exit: bool,
    /// Whether to enable mouse event capture.
    // TODO: Implement mouse capture
    pub capture_mouse: bool,
    /// Limits how frequently frames can be rendered. When set to 0, there is effectively no limit.
    pub min_frame_duration: Duration,
    /// Reduces CPU usage by limiting the frequency of each loop iteration.
    pub min_epoch_duration: Duration,
    /// Type of EventReciever to use for the application.
    pub event_reciever_type: EventRecieverType,
}

impl Default for ApplicationControlSettings {
    fn default() -> Self {
        ApplicationControlSettings {
            ctrl_c_exit: true,
            capture_mouse: false,
            min_frame_duration: Duration::from_secs(1) / 60,
            min_epoch_duration: Duration::from_secs(1) / 120,
            event_reciever_type: EventRecieverType::Threaded,
        }
    }
}

impl ApplicationControlSettings {
    /// Sets the minimum epoch duration based off of a maximum epochs per second.
    pub fn set_max_eps(&mut self, max_eps: u32) {
        self.min_epoch_duration = Duration::from_secs(1) / max_eps;
    }
    /// Sets the minimum frame duration based off of a maximum frames per second.
    pub fn set_max_fps(&mut self, max_fps: u32) {
        self.min_frame_duration = Duration::from_secs(1) / max_fps;
    }
}
