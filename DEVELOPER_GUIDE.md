# Developer Guide

This document serves as the developer guide for the SAGE (Sim2Real Actuator Gap Estimator) project, covering development workflows, code standards, and important development information.

## 🚀 Development Workflow

### 1. Branch Management

#### Creating New Branches
```bash
# Create new branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

#### Branch Naming Conventions
Use the following formats for branch names:

- **Feature Development**: `feature/feature-name`
  - Example: `feature/add-motion-analysis`
  - Example: `feature/improve-simulation-performance`

- **Bug Fixes**: `fix/bug-description`
  - Example: `fix/angle-conversion-error`
  - Example: `fix/memory-leak-in-simulation`

- **Refactoring**: `refactor/refactor-description`
  - Example: `refactor/simplify-data-processing`

- **Documentation**: `docs/documentation-update`
  - Example: `docs/update-api-documentation`

- **Hotfixes**: `hotfix/critical-fix`
  - Example: `hotfix/simulation-crash-fix`

### 2. Submitting Pull Requests

#### Pre-submission Checklist
- [ ] Code passes all tests
- [ ] Pre-commit hooks have been run
- [ ] Code follows Google Python Style Guide
- [ ] Added necessary documentation and comments
- [ ] Updated relevant README or documentation

#### PR Title Format
```
[Type] Brief description

Examples:
[Feature] Add motion file angle conversion utility
[Fix] Resolve simulation memory leak issue
[Refactor] Simplify analysis data processing pipeline
[Docs] Update installation and usage instructions
```

#### PR Description Template
```markdown
## 📋 Changes
- Brief description of your changes

## 🎯 Purpose
- Explain why these changes are needed

## 🧪 Testing
- Describe how you tested these changes
- Include relevant test commands or steps

## 📚 Documentation
- List any documentation updates
- Include new configurations or usage methods

## ⚠️ Breaking Changes
- Detail any breaking changes if applicable
```

### 3. Merging to Main Branch

#### Rebase Workflow
Before merging, you **MUST** rebase to the latest main branch:

```bash
# 1. Switch to main branch and pull latest code
git checkout main
git pull origin main

# 2. Switch back to your feature branch
git checkout feature/your-feature-name

# 3. Rebase to latest main
git rebase main

# 4. If conflicts occur, resolve them and continue
git add .
git rebase --continue

# 5. Force push to remote branch (only do this on feature branches)
git push --force-with-lease origin feature/your-feature-name
```

#### Merge Requirements
- ✅ Only PRs that pass all CI checks can be merged
- ✅ Requires at least one code review approval
- ✅ Must be rebased to latest main branch
- ✅ All discussions must be resolved

## 📝 Code Standards

### 1. Python Code Style

This project follows the **Google Python Style Guide**:
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

#### Key Style Guidelines

**Function and Variable Naming**
```python
# ✅ Good naming
def calculate_joint_angles():
    motion_file_path = "path/to/motion.txt"
    joint_angle_data = load_motion_data(motion_file_path)

# ❌ Avoid this naming
def calcJointAngles():
    motionFilePath = "path/to/motion.txt"
    jointAngleData = loadMotionData(motionFilePath)
```

**Class Naming**
```python
# ✅ Use PascalCase
class JointMotionAnalyzer:
    pass

class RobotSimulator:
    pass
```

**Constant Naming**
```python
# ✅ Use UPPER_SNAKE_CASE
MAX_JOINT_VELOCITY = 10.0
DEFAULT_SIMULATION_TIMESTEP = 0.01
```

**Import Order**
```python
# 1. Standard library imports
import os
import sys
from pathlib import Path

# 2. Related third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3. Local application/library imports
from sage.simulation import JointMotionBenchmark
from sage.analysis import RobotDataProcessor
```

**Docstrings**
```python
def convert_degrees_to_radians(degrees: float) -> float:
    """Convert angle from degrees to radians.

    Args:
        degrees: Angle in degrees.

    Returns:
        Angle in radians.

    Raises:
        ValueError: If degrees is not a valid number.
    """
    if not isinstance(degrees, (int, float)):
        raise ValueError("Input must be a number")
    return degrees * np.pi / 180.0
```

### 2. Pre-commit Hooks

#### Installation and Configuration
The project has pre-commit hooks configured. You **MUST** run checks before each commit:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install git hooks
pre-commit install

# Manually run all checks
pre-commit run --all-files
```

#### Pre-commit Checks
- **black**: Python code formatting
- **isort**: Import statement sorting
- **flake8**: Code style and error checking
- **mypy**: Type checking (if applicable)
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline

#### Commit Workflow
```bash
# 1. Add your changes
git add .

# 2. Commit (pre-commit will run automatically)
git commit -m "your commit message"

# 3. If pre-commit fails, fix issues and re-commit
# pre-commit may auto-fix some issues, requiring re-add and commit
git add .
git commit -m "your commit message"
```


## ❓ Frequently Asked Questions

### Q: What should I do if pre-commit checks fail?
A: Review the error messages, fix the issues, and re-commit. Many formatting issues are auto-fixed by pre-commit.

### Q: How do I resolve rebase conflicts?
A:
1. Manually resolve conflict files
2. `git add .`
3. `git rebase --continue`
4. Repeat until rebase is complete

### Q: I forgot the branch naming convention, what should I do?
A: Use `git branch -m old-name new-name` to rename your branch

---

**Happy Coding! 🎉**

For questions or suggestions, please submit an Issue or contact the project maintainers.
