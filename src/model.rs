use std::{
    any::{Any, TypeId},
    collections::BTreeMap,
};

use crossterm::event::Event;

use crate::message::{Message, MessageHandle};

pub struct ModelInstance<InnerState> {
    /// Internal state for the model.
    state: InnerState,
    /// Handles for different message types.
    message_handles: BTreeMap<TypeId, MessageHandle<InnerState>>,
    /// Handles recieving events
    event_handle: Box<dyn Fn(&mut InnerState, Event)>,
}

impl<InnerState> ModelInstance<InnerState> {
    /// Creates new ModelInstance
    pub fn new<F: Fn(&mut InnerState, Event) + 'static>(
        state: InnerState,
        event_handle: F,
    ) -> ModelInstance<InnerState> {
        return ModelInstance {
            state,
            message_handles: BTreeMap::new(),
            event_handle: Box::new(move |st, event| event_handle(st, event)),
        };
    }

    /// Registers a handle for a message type.
    pub fn register_message_handle<
        MessageType: Any,
        F: Fn(&mut InnerState, MessageType) + 'static,
    >(
        &mut self,
        handle: F,
    ) {
        let message_handle = MessageHandle::new(handle);
        self.message_handles
            .insert(message_handle.message_typeid, message_handle);
    }

    /// Finds the appropriate handle for a message if exists, and executes the handle.
    pub fn handle_message(&mut self, message: Message) {
        if let Some(handle) = self.message_handles.get(&message.contained_type()) {
            handle.handle_message(&mut self.state, message);
        }
    }

    /// Handles Events from crossterm.
    pub fn handle_event(&mut self, event: Event) {
        (*self.event_handle)(&mut self.state, event);
    }
}

/// Helper trait that can be implementeed on the struct used for the state in a ModelInstance
/// for recieving a Message.
pub trait HandlesMessage<MessageType> {
    fn handle_message(&mut self, message: MessageType);
}

/// Helper trait that can be implemented on the struct used for the state in a ModelInstance.
pub trait Model: Sized {
    fn handle_event(&mut self, event: Event);
}

pub(crate) trait ModelConstructor: Sized + Model {
    fn construct_model(self) -> ModelInstance<Self>;
}

#[macro_export]
macro_rules! build_model {
    ($st:expr $(,$message_type:ty)*) => {{
        use brass::model::ModelInstance;
        let mut instance = ModelInstance::new($st, |st, event| st.handle_event(event));
        $(
            instance.register_message_handle(|st, message: $message_type| st.handle_message(message));
        )*
        instance
    }};
}

pub use build_model;
