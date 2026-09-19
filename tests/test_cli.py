import json

from quantguard.cli import build_parser, main


def test_cli_verify_human_output(tmp_path, capsys):
    answer = tmp_path / "answer.txt"
    answer.write_text("Latency was 18ms.")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps(["Latency achieved was 12.4ms."]))

    exit_code = main(["verify", str(answer), "--sources", str(sources)])
    captured = capsys.readouterr()

    assert exit_code == 0  # no --fail-on-invalid, so exit 0 regardless of validity
    assert "CONTRADICTED" in captured.out
    assert "Overall valid: False" in captured.out


def test_cli_verify_json_output(tmp_path, capsys):
    answer = tmp_path / "answer.txt"
    answer.write_text("Latency was 12ms.")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps(["Latency achieved was 12.4ms."]))

    main(["verify", str(answer), "--sources", str(sources), "--json"])
    captured = capsys.readouterr()

    parsed = json.loads(captured.out)
    assert parsed["is_valid"] is True
    assert parsed["results"][0]["status"] == "VERIFIED"


def test_cli_verify_fail_on_invalid_exit_code(tmp_path):
    answer = tmp_path / "answer.txt"
    answer.write_text("Latency was 18ms.")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps(["Latency achieved was 12.4ms."]))

    exit_code = main(["verify", str(answer), "--sources", str(sources), "--fail-on-invalid"])
    assert exit_code == 1


def test_cli_sources_accepts_object_with_sources_key(tmp_path):
    answer = tmp_path / "answer.txt"
    answer.write_text("Latency was 12ms.")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"sources": ["Latency achieved was 12.4ms."]}))

    exit_code = main(["verify", str(answer), "--sources", str(sources), "--fail-on-invalid"])
    assert exit_code == 0


def test_cli_benchmark_command(tmp_path, capsys):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {"id": "case1", "generated_text": "Latency was 12ms.", "source_chunks": ["Latency achieved was 12.4ms."], "expected_valid": True},
                {"id": "case2", "generated_text": "Latency was 18ms.", "source_chunks": ["Latency achieved was 12.4ms."], "expected_valid": False},
            ]
        )
    )
    exit_code = main(["benchmark", str(dataset)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Accuracy: 2/2 = 100.0%" in captured.out


def test_parser_requires_a_subcommand():
    parser = build_parser()
    import pytest

    with pytest.raises(SystemExit):
        parser.parse_args([])
