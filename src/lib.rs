use std::{
    io::{self, stdout},
    thread::sleep,
    time::Instant,
};

use control_settings::ApplicationControlSettings;
use crossterm::{
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use event_reader::{EventReciever, NullEventReciever, ThreadedEventReciever};
use ratatui::{backend::CrosstermBackend, Terminal};

pub mod control_settings;
mod event_reader;
pub mod message;
pub mod model;

pub struct ApplicationController {
    pub control_settings: ApplicationControlSettings,
}

impl Default for ApplicationController {
    fn default() -> Self {
        ApplicationController {
            control_settings: ApplicationControlSettings::default(),
        }
    }
}

impl ApplicationController {
    /// Initialize the appropriate EventReciever instance.
    fn init_event_reciever(&self) -> Box<dyn EventReciever> {
        return match self.control_settings.event_reciever_type {
            control_settings::EventRecieverType::Null => Box::new(NullEventReciever),
            control_settings::EventRecieverType::Threaded => Box::new(ThreadedEventReciever::new()),
        };
    }

    pub fn start(&mut self) -> io::Result<()> {
        // Raw mode is needed to have direct access to input events
        stdout().execute(EnterAlternateScreen)?;
        enable_raw_mode()?;
        let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

        // Initialize event reciever
        let mut event_reciever = self.init_event_reciever();

        // Time last frame was dispatched
        let mut last_frame = Instant::now();

        loop {
            // Time this loop epoch started
            let epoch_start_instant = Instant::now();

            // Recieves queued inputs
            let event_queue = event_reciever.recieve();

            // Sleeps until next loop epoch time frame.
            let loop_time_elapsed = Instant::now() - epoch_start_instant;
            if let Some(sleep_time) = self
                .control_settings
                .min_epoch_duration
                .checked_sub(loop_time_elapsed)
            {
                sleep(sleep_time);
            }
        }

        // Cleanup terminal
        disable_raw_mode()?;
        stdout().execute(LeaveAlternateScreen)?;
        return Ok(());
    }
}
