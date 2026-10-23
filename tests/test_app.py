from streamlit.testing.v1 import AppTest


def run_app() -> AppTest:
    return AppTest.from_file("../app.py").run(timeout=30)


def number_input(at: AppTest, label: str):
    return next(w for w in at.sidebar.number_input if w.label == label)


def test_default_beam_is_safe():
    at = run_app()

    assert not at.exception
    assert "Safe." in at.success[0].value


def test_shortened_span_flags_overhanging_load():
    at = run_app()
    number_input(at, "Span (m)").set_value(5.0).run()

    assert not at.exception
    assert any("runs past the beam" in w.value for w in at.warning)


def test_thin_wood_beam_is_failing():
    at = run_app()
    at.sidebar.selectbox[0].select("Douglas fir").run()
    number_input(at, "Height (mm)").set_value(60.0).run()

    assert not at.exception
    assert "Failing." in at.error[0].value
