# Build Configuration

Build scripts and configurations for creating installers.

## Structure

```
build/
├── windows/                    # Windows build files
│   ├── classmark.spec         # PyInstaller specification
│   ├── build_windows.bat      # Build executable script
│   ├── build_installer.bat    # Build installer script
│   └── classmark_installer.iss # Inno Setup configuration
└── macos/                      # macOS build files
    ├── setup_macos.py         # py2app configuration
    ├── build_macos.sh         # Build app bundle script
    └── build_dmg.sh           # Create DMG script
```

## Building for Windows

From project root:

```cmd
cd build\windows
build_windows.bat
build_installer.bat
```

**Output:**
- `../../dist/Classmark/Classmark.exe`
- `../../Output/ClassmarkSetup.exe`

## Building for macOS

From project root:

```bash
cd build/macos
./build_macos.sh
./build_dmg.sh
```

**Output:**
- `../../dist/Classmark.app`
- `../../Output/Classmark-1.0.0.dmg`

## Prerequisites

See BUILD_INSTRUCTIONS.md in docs/ for detailed prerequisites.

## Notes

- Build scripts expect to be run from their respective directories
- Adjust paths in build scripts if moving files
- See individual README files in windows/ and macos/ for details
