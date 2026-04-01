#pragma once

// Forward declarations — keeps this header lean.
namespace httplib { class Server; }
namespace engram  { class SessionStore; }

namespace engram {

/// Register all HTTP routes on `srv` backed by `store`.
/// Call once before srv.listen().
void register_routes(httplib::Server& srv, SessionStore& store);

} // namespace engram
