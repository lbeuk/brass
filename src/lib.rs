use control_settings::ApplicationControlSettings;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use model_tree::ModelTreeNode;
use ratatui::{backend::CrosstermBackend, Frame, Terminal};
use std::{
    cmp::Reverse,
    collections::VecDeque,
    io::{self, stdout},
    sync::mpsc::channel,
    thread::{self, sleep},
    time::Instant,
};

pub mod control_settings;
pub mod model_tree;
/// ModelViews represent an MVC model.
pub trait ModelView {
    /// Custom update types used by application.
    type EventType;
    fn render_view(&mut self, frame: &mut Frame);
}

/// Helper function to check for a Ctrl + C press
pub fn check_ctrl_c(event: &Event) -> bool {
    let Event::Key(key_event) = event else {
        return false;
    };

    if key_event.kind != KeyEventKind::Press {
        return false;
    }

    if key_event.modifiers != KeyModifiers::CONTROL {
        return false;
    }

    if key_event.code != KeyCode::Char('c') {
        return false;
    };

    return true;
}

pub struct ApplicationController {
    pub control_settings: ApplicationControlSettings,
    pub model_tree: Option<ModelTreeNode>,
}

impl Default for ApplicationController {
    fn default() -> Self {
        ApplicationController {
            control_settings: ApplicationControlSettings::default(),
            model_tree: None,
        }
    }
}

impl ApplicationController {
    pub fn start(&mut self) -> io::Result<()> {
        // Raw mode is needed to have direct access to input events
        stdout().execute(EnterAlternateScreen)?;
        enable_raw_mode()?;
        let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

        // Create input management thread.
        let (tx, rx) = channel();
        thread::spawn(move || loop {
            // TODO: Handle more gracefully than unwrap
            let input_event = event::read().unwrap();
            tx.send(input_event).unwrap();
        });

        let mut last_frame = Instant::now();

        'epoch_loop: loop {
            let epoch_start_instant = Instant::now();

            let mut event_deque = VecDeque::new();

            // Recieves queued inputs
            while let Ok(input_event) = rx.try_recv() {
                // Handles exiting on Ctrl + C press, otherwise queueing the event.
                if self.control_settings.ctrl_c_exit && check_ctrl_c(&input_event) {
                    break 'epoch_loop;
                } else {
                    event_deque.push_back(UpdateEvent::Input(input_event));
                }
            }

            // Dispatched events
            while let (Some(view), Some(update_event)) = (&mut self.view, event_deque.pop_front()) {
                let feedback = view.update_model(update_event);
                if feedback.exit_signal {
                    break 'epoch_loop;
                }
                for scheduled_event in feedback.schedule_queue {
                    self.event_queue.push(Reverse(scheduled_event));
                }
            }

            // Call frame update if appropriate interval has elapsed.
            let now = Instant::now();
            let elapsed = now - last_frame;
            if elapsed > self.control_settings.min_frame_duration {
                last_frame = now;
                if let Some(view) = &mut self.view {
                    terminal.draw(|f| view.render_view(f))?;
                };
            }

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

        disable_raw_mode()?;
        stdout().execute(LeaveAlternateScreen)?;
        return Ok(());
    }

    pub fn set_view<V: ModelView<EventType = T> + 'static>(&mut self, view: V) {
        self.view = Some(Box::new(view));
    }
}
