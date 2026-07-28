//! 请求转发器
//!
//! 负责将请求转发到上游Provider，支持故障转移

use super::hyper_client::ProxyResponse;
use super::{
    body_filter::filter_private_params_with_whitelist,
    content_encoding::{decompress_body, get_content_encoding},
    error::*,
    failover_switch::FailoverSwitchManager,
    json_canonical::{canonicalize_value, short_value_hash},
    log_codes::fwd as log_fwd,
    provider_router::ProviderRouter,
    providers::{
        codex_chat_history::CodexChatHistoryStore, gemini_shadow::GeminiShadowStore, get_adapter,
        AuthInfo, AuthStrategy, ProviderAdapter, ProviderType,
    },
    thinking_budget_rectifier::{rectify_thinking_budget, should_rectify_thinking_budget},
    thinking_rectifier::{
        normalize_thinking_type, rectify_anthropic_request, should_rectify_thinking_signature,
    },
    types::{CopilotOptimizerConfig, OptimizerConfig, ProxyStatus, RectifierConfig},
    ProxyError,
};
use crate::commands::{CodexOAuthState, CopilotAuthState, XaiOAuthState};
use crate::proxy::providers::codex_oauth_auth::CodexOAuthManager;
use crate::proxy::providers::copilot_auth::CopilotAuthManager;
use crate::proxy::providers::xai_oauth_auth::XaiOAuthManager;
use crate::{
    app_config::AppType,
    provider::{LocalProxyRequestOverrides, Provider},
};
use bytes::Bytes;
use futures::StreamExt;
use http::Extensions;
use serde_json::Value;
use std::sync::Arc;
use tauri::Manager;
use tokio::sync::RwLock;

const PROXY_AUTH_PLACEHOLDER: &str = "PROXY_MANAGED";