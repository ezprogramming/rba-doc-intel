from app.rag.hooks import HookBus


def test_hook_bus_invokes_event_specific_handlers():
    bus = HookBus()
    captured = []

    def handler(event, payload):
        captured.append((event, payload))

    bus.subscribe("rag:event", handler)
    bus.emit("rag:event", foo="bar")

    assert captured == [("rag:event", {"foo": "bar"})]


def test_hook_bus_global_handlers_receive_all_events():
    bus = HookBus()
    captured = []

    def handler(event, payload):
        captured.append(event)

    bus.subscribe_all(handler)
    bus.emit("a:event")
    bus.emit("b:event")

    assert captured == ["a:event", "b:event"]
