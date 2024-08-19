# Install
pip3 install -r requirements.txt


# Run
Example run:

```bash
python3 -m src.run_eval --providers-path providers.json --settings-path settings.json --output-path results/en/claude_3_haiku.json  --testee-name claude-3-haiku  --tester-name claude-3-5-sonnet --language en
```

# Run Jekyll locally

```bash
cd pages
bundle exec jekyll serve --host 127.0.0.1 --port 8000
```


# Contribute

## Linting
```
pip3 install mypy flake8 black
flake8 src
black src --line-length 100
mypy src --strict
```
