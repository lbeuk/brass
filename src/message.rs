use std::{
    any::{Any, TypeId},
    time::{Duration, Instant},
};

/// Schedule for when a Message should be dispatched.
pub enum MessageSchedule {
    /// Scheduled for the following epoch.
    NextEpoch,
    /// Scheduled for a given instant.
    Instant(Instant),
}

impl MessageSchedule {
    /// Helper function for constructing MessageSchedule::Instant instances
    /// with a given offset duration.
    pub fn in_duration(duration: Duration) -> MessageSchedule {
        return MessageSchedule::Instant(Instant::now() + duration);
    }
}

/// Empty type representing a message that is just a ping.
pub struct PingMessage;

/// Message passed between models.
pub struct Message {
    /// Data contained by message.
    data: Box<dyn Any>,
    /// When the message is scheduled to be delivered to the destination model.
    pub scheduled_for: MessageSchedule,
}

impl Message {
    /// Create a new message.
    pub fn new<T: Any>(message: T, scheduled_for: MessageSchedule) -> Message {
        Message {
            data: Box::new(message),
            scheduled_for,
        }
    }

    /// Creates a new ping message. Ping messages are self-addressed, empty messages,
    /// and can be used when you want to update the model in the next epoch. A ping
    /// can be sent on every epoch if it is desired that the model is updated every epoch.
    pub fn new_ping() -> Message {
        Message {
            data: Box::new(PingMessage),
            scheduled_for: MessageSchedule::NextEpoch,
        }
    }

    /// Creates a new ping message, except with a delay.
    pub fn new_delayed_ping(duration: Duration) -> Message {
        Message {
            data: Box::new(PingMessage),
            scheduled_for: MessageSchedule::in_duration(duration),
        }
    }

    /// Returns TypeId of the message.
    pub fn contained_type(&self) -> TypeId {
        return (*self.data).type_id();
    }
}

/// Represents a closure to handle to mutate a specified state type (InnerState) from
/// a specified message type.
pub(crate) struct MessageHandle<InnerState> {
    handle: Box<dyn Fn(&mut InnerState, Box<dyn Any>)>,
    pub(crate) message_typeid: TypeId,
}

impl<InnerState> MessageHandle<InnerState> {
    /// Generates a new MessageHandle from a passed in function/closure.
    pub(crate) fn new<MessageType: Any, F: Fn(&mut InnerState, MessageType) + 'static>(
        handle: F,
    ) -> MessageHandle<InnerState> {
        MessageHandle {
            handle: Box::new(move |st, message| {
                // Ideally, this unwrap should never fail since it is guarded by the TypeId check
                // in handle_message.
                handle(st, *message.downcast::<MessageType>().unwrap())
            }),
            message_typeid: TypeId::of::<MessageType>(),
        }
    }

    pub(crate) fn handle_message(&self, state: &mut InnerState, message: Message) {
        // TODO: Handle else condition.
        if message.contained_type() == self.message_typeid {
            (self.handle)(state, message.data);
        }
    }
}
