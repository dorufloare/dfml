#pragma once

namespace dfml {

inline thread_local bool grad_enabled = true;

struct GradGuard {
    bool old_state;
    GradGuard(bool enabled=true) : old_state(grad_enabled) { grad_enabled = enabled; }
    ~GradGuard() { grad_enabled = old_state; }

    static bool is_grad_enabled() { return grad_enabled; }
};

} // namespace dfml