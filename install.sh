#!/bin/bash
################################################################################
# Classmark: SOTA Classification Marking Detection System
# Installation Script
#
# This script installs all dependencies and sets up the environment for
# the Classmark classification marking detection system.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh [--gpu] [--minimal] [--dev]
#
# Options:
#   --gpu      Install GPU support (CUDA, cuDNN)
#   --minimal  Install only core dependencies (no ML/AI features)
#   --dev      Install development dependencies (testing, linting)
#
# Author: Classmark Development Team
# Date: 2025-11-10
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.10"
GPU_SUPPORT=false
MINIMAL_INSTALL=false
DEV_INSTALL=false

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --gpu)
            GPU_SUPPORT=true
            shift
            ;;
        --minimal)
            MINIMAL_INSTALL=true
            shift
            ;;
        --dev)
            DEV_INSTALL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--gpu] [--minimal] [--dev]"
            echo ""
            echo "Options:"
            echo "  --gpu      Install GPU support (CUDA, cuDNN)"
            echo "  --minimal  Install only core dependencies"
            echo "  --dev      Install development dependencies"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print functions
print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        echo "Please install Python 3.10+ and try again"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

    if [ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$PYTHON_MIN_VERSION" ]; then
        print_error "Python $PYTHON_VERSION is installed, but $PYTHON_MIN_VERSION or higher is required"
        exit 1
    fi

    print_success "Python $PYTHON_VERSION detected"
}

# Install system dependencies
install_system_dependencies() {
    print_header "Installing System Dependencies"

    if command_exists apt-get; then
        # Debian/Ubuntu
        print_info "Detected Debian/Ubuntu system"

        sudo apt-get update
        print_success "Package list updated"

        # Core dependencies
        sudo apt-get install -y tesseract-ocr poppler-utils
        print_success "Installed tesseract-ocr and poppler-utils"

        if [ "$GPU_SUPPORT" = true ]; then
            print_info "GPU support requested - checking for NVIDIA drivers..."

            if command_exists nvidia-smi; then
                print_success "NVIDIA drivers detected"
                nvidia-smi | head -n 3
            else
                print_warning "NVIDIA drivers not found - GPU features will not work"
                print_info "Install NVIDIA drivers: sudo apt-get install nvidia-driver-XXX"
            fi
        fi

    elif command_exists yum; then
        # RedHat/CentOS/Fedora
        print_info "Detected RedHat/CentOS/Fedora system"

        sudo yum install -y tesseract poppler-utils
        print_success "Installed tesseract and poppler-utils"

    elif command_exists brew; then
        # macOS
        print_info "Detected macOS system"

        brew install tesseract poppler
        print_success "Installed tesseract and poppler"

    else
        print_warning "Unknown package manager - please install manually:"
        echo "  - tesseract-ocr"
        echo "  - poppler-utils"
    fi
}

# Install Python dependencies
install_python_dependencies() {
    print_header "Installing Python Dependencies"

    # Upgrade pip
    python3 -m pip install --upgrade pip
    print_success "pip upgraded"

    if [ "$MINIMAL_INSTALL" = true ]; then
        print_info "Minimal installation - installing only core dependencies"

        # Install only core dependencies
        python3 -m pip install \
            PyPDF2>=3.0.0 \
            python-docx>=1.0.0 \
            pdfplumber>=0.10.0 \
            docx2python>=2.0.0 \
            rapidfuzz>=3.0.0 \
            tqdm>=4.65.0

        print_success "Core dependencies installed"

    else
        print_info "Full installation - installing all dependencies"

        # Install from requirements.txt
        if [ -f "requirements.txt" ]; then
            python3 -m pip install -r requirements.txt
            print_success "All dependencies installed from requirements.txt"
        else
            print_error "requirements.txt not found"
            exit 1
        fi
    fi

    if [ "$DEV_INSTALL" = true ]; then
        print_info "Installing development dependencies"

        python3 -m pip install \
            pytest>=7.0.0 \
            pytest-cov>=4.0.0 \
            black>=23.0.0 \
            flake8>=6.0.0 \
            mypy>=1.0.0

        print_success "Development dependencies installed"
    fi
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"

    # Check Python imports
    print_info "Checking Python imports..."

    python3 -c "import PyPDF2" && print_success "PyPDF2 imported" || print_error "PyPDF2 import failed"
    python3 -c "import docx" && print_success "python-docx imported" || print_error "python-docx import failed"

    if [ "$MINIMAL_INSTALL" != true ]; then
        python3 -c "import torch" && print_success "PyTorch imported" || print_warning "PyTorch import failed (GPU features unavailable)"
        python3 -c "import transformers" && print_success "Transformers imported" || print_warning "Transformers import failed (LayoutLM unavailable)"
        python3 -c "import ahocorasick" && print_success "pyahocorasick imported" || print_warning "pyahocorasick import failed (Fast matching unavailable)"
        python3 -c "import sklearn" && print_success "scikit-learn imported" || print_warning "scikit-learn import failed (Evaluation unavailable)"
    fi

    # Check system commands
    print_info "Checking system commands..."

    command_exists tesseract && print_success "Tesseract OCR available" || print_warning "Tesseract OCR not found"
    command_exists pdftoppm && print_success "pdftoppm available" || print_warning "pdftoppm not found (poppler-utils)"

    if [ "$GPU_SUPPORT" = true ]; then
        command_exists nvidia-smi && print_success "NVIDIA GPU available" || print_warning "NVIDIA GPU not detected"
    fi
}

# Run tests
run_tests() {
    print_header "Running Tests"

    if [ -f "test_fast_pattern_matcher.py" ]; then
        print_info "Running pattern matcher tests..."
        python3 test_fast_pattern_matcher.py > /dev/null 2>&1 && \
            print_success "Pattern matcher tests passed" || \
            print_warning "Pattern matcher tests failed (check dependencies)"
    fi

    if [ -f "test_hybrid_classifier.py" ] && [ "$MINIMAL_INSTALL" != true ]; then
        print_info "Running hybrid classifier tests..."
        python3 test_hybrid_classifier.py > /dev/null 2>&1 && \
            print_success "Hybrid classifier tests passed" || \
            print_warning "Hybrid classifier tests failed (check GPU/dependencies)"
    fi
}

# Print summary
print_summary() {
    print_header "Installation Summary"

    echo "Installation completed with the following configuration:"
    echo ""
    echo "  Python Version: $(python3 --version | cut -d' ' -f2)"
    echo "  GPU Support: $([ "$GPU_SUPPORT" = true ] && echo "Enabled" || echo "Disabled")"
    echo "  Installation Type: $([ "$MINIMAL_INSTALL" = true ] && echo "Minimal" || echo "Full")"
    echo "  Development Mode: $([ "$DEV_INSTALL" = true ] && echo "Enabled" || echo "Disabled")"
    echo ""

    print_success "Classmark is ready to use!"
    echo ""
    echo "Quick Start:"
    echo "  1. Run baseline evaluation:"
    echo "     python3 run_baseline_evaluation.py"
    echo ""
    echo "  2. Test hybrid classifier:"
    echo "     python3 test_hybrid_classifier.py"
    echo ""
    echo "  3. Use the scanner:"
    echo "     python3 -c \"from classification_scanner import ClassificationScanner; scanner = ClassificationScanner(); print(scanner.scan_file('document.pdf'))\""
    echo ""
    echo "For more information, see:"
    echo "  - README.md"
    echo "  - PROJECT_SUMMARY.md"
    echo "  - PHASE3_SUMMARY.md"
    echo "  - PHASE4_SUMMARY.md"
}

# Main installation flow
main() {
    print_header "Classmark Installation Script"

    echo "This script will install all dependencies for the Classmark"
    echo "classification marking detection system."
    echo ""
    echo "Configuration:"
    echo "  GPU Support: $([ "$GPU_SUPPORT" = true ] && echo "YES" || echo "NO")"
    echo "  Minimal Install: $([ "$MINIMAL_INSTALL" = true ] && echo "YES" || echo "NO")"
    echo "  Development Mode: $([ "$DEV_INSTALL" = true ] && echo "YES" || echo "NO")"
    echo ""

    read -p "Continue with installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled"
        exit 0
    fi

    # Run installation steps
    check_python
    install_system_dependencies
    install_python_dependencies
    verify_installation

    # Optional: Run tests
    if [ "$DEV_INSTALL" = true ]; then
        read -p "Run tests? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_tests
        fi
    fi

    print_summary
}

# Run main function
main "$@"
