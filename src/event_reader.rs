use std::{
    collections::VecDeque,
    sync::mpsc::{channel, Receiver},
    thread,
};

use crossterm::event::{self, Event};

/// Trait defining a structure that helps recieve crossterm events,
pub trait EventReciever {
    /// Recieve the queued events.
    fn recieve(&mut self) -> VecDeque<Event>;
}

/// EventReciever that ignored all events.
pub struct NullEventReciever;

impl EventReciever for NullEventReciever {
    fn recieve(&mut self) -> VecDeque<Event> {
        return VecDeque::new();
    }
}

/// EventReciever that offloads reading to a thread.
pub struct ThreadedEventReciever {
    rx: Receiver<Event>,
}

impl EventReciever for ThreadedEventReciever {
    fn recieve(&mut self) -> VecDeque<Event> {
        return VecDeque::from_iter(self.rx.try_iter());
    }
}

impl ThreadedEventReciever {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        thread::spawn(move || loop {
            let input_event = event::read().unwrap();
            tx.send(input_event).unwrap();
        });
        return ThreadedEventReciever { rx };
    }
}
