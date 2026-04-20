// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

pub type Edge = (String, String, f64);

// pub use self::network::Network;
// pub use self::network_builder::NetworkBuilder;

pub use self::compact_network::CompactNodeId;
pub use self::identifier::Identifier;
pub use self::labeled_network::{LabeledNetwork, LabeledNetworkBuilder};

mod compact_network;
mod identifier;
mod labeled_network;
mod networks;
pub mod prelude;
