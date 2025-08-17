# Contributing to DarwinTD

## Development Workflow

### Feature Branch Strategy

Starting immediately, we use a feature branch workflow:

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/description-of-feature
   # or for specific components:
   git checkout -b setup-detection/fibonacci-v3
   git checkout -b quality-eval/statistical-v1  
   git checkout -b execution/adaptive-v1
   git checkout -b orchestration/genetic-optimizer
   ```

2. **Work on Feature**
   ```bash
   # Make your changes
   git add .
   git commit -m "implement: specific feature description"
   git push origin feature/your-branch-name
   ```

3. **Create Pull Request**
   - Use the PR template in `.github/pull_request_template.md`
   - Link to related GitHub issues
   - Include testing and performance information

4. **Code Review & Merge**
   - Address review feedback
   - Ensure all checks pass
   - Merge to main branch

### Branch Naming Conventions

```
feature/general-description
setup-detection/engine-name-version
quality-eval/evaluator-name-version  
execution/executor-name-version
orchestration/component-name
bugfix/issue-description
docs/documentation-update
```

## Modular Architecture Guidelines

### Adding New Setup Detection Engines

1. Create in `/src/darwintd/setup_detection/[engine_type]/`
2. Inherit from `BaseSetupDetector`
3. Implement required methods:
   ```python
   def detect_setups(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> List[SetupData]
   def get_parameter_ranges(self) -> Dict[str, tuple]
   ```
4. Add comprehensive tests
5. Update orchestration registration

### Adding New Quality Evaluation Engines

1. Create in `/src/darwintd/quality_evaluation/[eval_type]/`
2. Inherit from `BaseQualityEvaluator`
3. Implement required methods:
   ```python
   def evaluate_setup(self, setup: SetupData, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> QualityScore
   def get_parameter_ranges(self) -> Dict[str, tuple]
   ```

### Adding New Trade Execution Engines

1. Create in `/src/darwintd/trade_execution/[exec_type]/`
2. Inherit from `BaseTradeExecutor`
3. Implement required methods:
   ```python
   def execute_setup(self, setup: SetupData, quality: QualityScore, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Optional[TradeExecution]
   def manage_open_trades(self, market_data: pd.DataFrame, current_time: datetime) -> List[TradeExecution]
   def get_parameter_ranges(self) -> Dict[str, tuple]
   ```

## Code Quality Standards

### Testing Requirements
- Unit tests for all new components
- Integration tests for pipeline changes
- Performance benchmarks for optimization components
- Visual validation for setup detection engines

### Documentation Standards
- Docstrings for all public methods
- Type hints for all function signatures
- Usage examples in module docstrings
- Update CLAUDE.md for architectural changes

### Performance Requirements
- Setup detection: Process 1+ years hourly data in <30 seconds
- Quality evaluation: <100ms per setup evaluation
- Trade execution: Real-time operation capability
- Backtesting: Maintain >100 backtests/second

## Commit Message Format

```
type: brief description

Longer description if needed explaining the what and why.

- Key changes made
- Components affected
- Performance implications

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Commit Types
- `feat:` New feature or engine
- `fix:` Bug fix
- `perf:` Performance improvement
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Engine Versioning Strategy

### Version Numbering
- `v1_basic`: Initial implementation with core functionality
- `v2_advanced`: Enhanced with additional features/optimizations
- `v3_optimized`: Performance-optimized or ML-enhanced version

### Backward Compatibility
- All engine versions must maintain interface compatibility
- New parameters should have sensible defaults
- Deprecated engines marked clearly before removal

## Development Environment

### Required Tools
- Python 3.11+
- Docker for consistent environment
- Git for version control
- Your preferred IDE with Python support

### Setup Commands
```bash
# Clone and setup
git clone https://github.com/martinjms/darwintd.git
cd darwintd
docker-compose up -d  # Start development environment

# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# Work on feature, test, commit, push, create PR
```

## Issue Tracking

### Creating Issues
- Use appropriate labels (setup-detection, quality-eval, execution, etc.)
- Include clear acceptance criteria
- Link to related architectural components
- Provide implementation examples when helpful

### Issue Labels
- `setup-detection`: Setup detection engines
- `quality-eval`: Quality evaluation engines
- `execution`: Trade execution engines
- `orchestration`: Pipeline and optimization
- `performance`: Performance improvements
- `documentation`: Documentation updates
- `bug`: Bug reports
- `enhancement`: Feature improvements

## Questions?

For questions about contributing:
1. Check existing issues and documentation
2. Create a new issue with the `question` label
3. Reference relevant architecture components

Thank you for contributing to DarwinTD! 🚀