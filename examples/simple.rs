use brass::{
    message::{Message, PingMessage},
    model::{build_model, HandlesMessage, Model},
};
use crossterm::event::Event;

struct Counter {
    ctr: usize,
}

impl Model for Counter {
    fn handle_event(&mut self, _event: Event) {
        return;
    }
}

impl HandlesMessage<usize> for Counter {
    fn handle_message(&mut self, message: usize) {
        self.ctr += message;
        println!("Counter: {}", self.ctr);
    }
}

impl HandlesMessage<PingMessage> for Counter {
    fn handle_message(&mut self, _message: PingMessage) {
        self.ctr += 1;
        println!("Counter: {}", self.ctr);
    }
}

impl Default for Counter {
    fn default() -> Self {
        Counter { ctr: 0 }
    }
}

fn main() {
    let mut model = build_model!(Counter::default(), usize, PingMessage);

    for _ in 0..4 {
        model.handle_message(Message::new_ping());
    }

    for i in 0..4 {
        model.handle_message(Message::new(
            i as usize,
            brass::message::MessageSchedule::NextEpoch,
        ));
    }
}
