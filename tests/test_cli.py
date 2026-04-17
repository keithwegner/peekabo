from typer.testing import CliRunner

from wifi_id.cli import app


def test_cli_help_smoke():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "eval-prequential" in result.output

