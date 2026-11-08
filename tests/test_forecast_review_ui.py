"""Review/download reruns retain advice and do not recompute the live model."""


def test_live_advice_survives_rerun_and_changing_settings_invalidates_it(monkeypatch,tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile,'tempdir',str(tmp_path))
    from streamlit.testing.v1 import AppTest
    import app
    from src.ingestion import live
    monkeypatch.setattr(live,'season_started',lambda:True)
    def build(*args,**kwargs):
        app.st.session_state['test_build_calls']=app.st.session_state.get('test_build_calls',0)+1
        return {'ok':True,'gw':11}
    monkeypatch.setattr(app,'build_advice',build)
    monkeypatch.setattr(app,'render_html',lambda advice:'<div>Saved advice remains visible</div>')
    script=tmp_path/'review_app.py'
    script.write_text('import app\napp.live_advice_view()\n',encoding='utf-8')
    at=AppTest.from_file(str(script)).run()
    assert not at.exception
    at.text_input[0].set_value('2942747').run()
    next(button for button in at.button if button.label=='Get advice').click().run()
    assert not at.exception
    assert at.session_state['test_build_calls']==1
    assert any('Saved advice remains visible' in element.value for element in at.markdown)
    at.run()
    assert at.session_state['test_build_calls']==1
    assert any('Saved advice remains visible' in element.value for element in at.markdown)
    at.checkbox[0].set_value(True).run()
    assert not any('Saved advice remains visible' in element.value for element in at.markdown)
    assert at.session_state['test_build_calls']==1


def test_forecast_review_initial_screen_renders_without_advice(monkeypatch,tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile,'tempdir',str(tmp_path))
    from streamlit.testing.v1 import AppTest
    script=tmp_path/'initial_review.py'
    script.write_text('import app\napp.forecast_review_controls()\n',encoding='utf-8')
    at=AppTest.from_file(str(script)).run()
    assert not at.exception
    compare=next(button for button in at.button if button.label=='Compare with checked results')
    assert compare.disabled
