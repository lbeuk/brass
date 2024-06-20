use crossterm::event::Event;
use rand;
use ratatui::{layout::Rect, Frame};
use std::{
    any::{Any, TypeId},
    collections::BTreeMap,
    rc::{Rc, Weak},
};
use uuid::Uuid;

pub trait Model {
    /// Type that the Model will recieve in recieve_message when updating internal state.
    type Message;

    /// Recieve a message and update internal state accordingly.
    fn recieve_message(&mut self, message: Self::Message);

    /// Recieve an event from Crossterm that has been routed to this component.
    fn recieve_event(&mut self, event: Event);

    /// Render the model to the screen.
    fn render(&mut self, frame: &mut Frame, recomended_region: Option<Rect>);
}

/// Tree holding the models.
pub struct ModelTreeNodeData {
    /// Model which the node refers to.
    node: Box<dyn Model<Message = dyn Any>>,
    /// UUID for the node.
    node_uuid: Uuid,
    /// Parent node, if set to None, that implies this node is the tree root.
    parent: Option<ModelTreeNodeRef>, // Weak prevents memory leak caused by cyclical Rc
    /// Map of children nodes, sorted by their node_uuid.
    children: BTreeMap<Uuid, ModelTreeNode>,
    /// TypeId for the EventType used by the Model.
    event_type_id: TypeId,
}

pub struct ModelTreeNode {
    pub node: Rc<ModelTreeNodeData>,
}

impl ModelTreeNode {
    /// Creates a new node from a given Model
    pub fn new<M: Model<Message = dyn Any + 'static>>(model: M) -> ModelTreeNode {
        let mut rng = rand::thread_rng();
        return ModelTreeNode {
            node: Rc::new(ModelTreeNodeData {
                node: Box::new(model),
                node_uuid: Uuid::new_v4(),
                parent: None,
                children: BTreeMap::new(),
                event_type_id: TypeId::of::<M::Message>(),
            }),
        };
    }

    /// Adds a Model as a child to the present Model
    pub fn add_child<M: Model<Message = dyn Any + 'static>>(&mut self, model: M) {
        let mut new_child = ModelTreeNode::new(model);
        self.node.parent = Some(ModelTreeNodeRef {
            node_ref: Rc::downgrade(&self.node),
        });
        self.node
            .children
            .insert(new_child.node.node_uuid, new_child);
    }

    /// Helper to aid in getting a ModelTreeNode for the parent node.
    pub fn get_parent(&self) -> Option<ModelTreeNode> {
        return Some(ModelTreeNode {
            node: self.node.parent?.node_ref.upgrade()?,
        });
    }
}

pub struct ModelTreeNodeRef {
    pub node_ref: Weak<ModelTreeNodeData>,
}
