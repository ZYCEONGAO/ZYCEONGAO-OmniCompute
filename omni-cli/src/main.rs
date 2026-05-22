//! # main
//!
//! Entry point for the `omnicompute` terminal toolchain.
//! Configures argument parsers and logging, handling user subcommands.

use crate::commands::CliCommands;
use clap::Parser;
use tracing_subscriber::EnvFilter;

pub mod commands;

/// Command line utility structure representing the OmniCompute toolchain.
#[derive(Parser, Debug)]
#[clap(
    name = "omnicompute",
    author = "Google DeepMind - Antigravity Agent & OmniCompute Contributors",
    version = "0.1.0",
    about = "Break the Silica Walls. Liquidize Global Compute. Free the AI Generation."
)]
struct Cli {
    #[clap(subcommand)]
    command: CliCommands,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize subscriber for elegant tracing outputs
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    cli.command.execute().await?;

    Ok(())
}
