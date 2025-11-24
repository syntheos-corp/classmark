; ============================================================================
; Classmark Windows Installer Script (Inno Setup)
;
; This script creates a professional Windows installer for Classmark
;
; Prerequisites:
;   1. Build the application first: build_windows.bat
;   2. Download models: python download_models.py
;   3. Install Inno Setup: https://jrsoftware.org/isdl.php
;
; Usage:
;   1. Open this file in Inno Setup Compiler
;   2. Click Build > Compile
;   OR
;   3. Run from command line: iscc classmark_installer.iss
;
; Output:
;   Output\ClassmarkSetup.exe
;
; Author: Classmark Development Team
; Date: 2025-11-10
; ============================================================================

#define MyAppName "Classmark"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Classmark Development Team"
#define MyAppURL "https://github.com/your-org/classmark"
#define MyAppExeName "Classmark.exe"

[Setup]
; App information
AppId={{A5B2C3D4-E5F6-4A5B-8C9D-0E1F2A3B4C5D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Installation directories
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Output
OutputDir=Output
OutputBaseFilename=ClassmarkSetup
Compression=lzma2/ultra64
SolidCompression=yes

; Privileges
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; Appearance
WizardStyle=modern
; SetupIconFile=classmark.ico
; WizardImageFile=wizard.bmp
; WizardSmallImageFile=wizard-small.bmp

; Architecture
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; License and readme
; LicenseFile=LICENSE.txt
; InfoBeforeFile=README.md

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application files
Source: "dist\Classmark\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Models directory (if present)
Source: "models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: ModelsExist

; Configuration
Source: "models\models_config.json"; DestDir: "{app}"; Flags: ignoreversion; Check: ConfigExists

; Documentation
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme; Check: FileExists('README.md')
Source: "PROJECT_SUMMARY.md"; DestDir: "{app}\docs"; Flags: ignoreversion; Check: FileExists('PROJECT_SUMMARY.md')
Source: "PHASE3_SUMMARY.md"; DestDir: "{app}\docs"; Flags: ignoreversion; Check: FileExists('PHASE3_SUMMARY.md')
Source: "PHASE4_SUMMARY.md"; DestDir: "{app}\docs"; Flags: ignoreversion; Check: FileExists('PHASE4_SUMMARY.md')

; System dependencies info
Source: "install.sh"; DestDir: "{app}\docs"; Flags: ignoreversion; Check: FileExists('install.sh')

[Icons]
; Start menu shortcuts
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

; Desktop shortcut
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Option to run the application after installation
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function ModelsExist: Boolean;
begin
  Result := DirExists(ExpandConstant('{src}\models'));
end;

function ConfigExists: Boolean;
begin
  Result := FileExists(ExpandConstant('{src}\models\models_config.json'));
end;

function FileExists(FileName: string): Boolean;
begin
  Result := FileOrDirExists(FileName);
end;

procedure InitializeWizard;
var
  ModelsPage: TOutputMsgMemoWizardPage;
  InfoText: string;
begin
  // Create custom page with information about models
  ModelsPage := CreateOutputMsgMemoPage(wpWelcome,
    'Important Information', 'AI Models Required',
    'Classmark uses AI models for classification detection. ' +
    'These models are required for the application to function.',
    '');

  if ModelsExist then
  begin
    InfoText := 'AI models found - will be installed' + #13#10 +
                'Total size: ~600 MB' + #13#10#13#10 +
                'After installation, Classmark will work completely offline.';
  end
  else
  begin
    InfoText := 'AI models NOT found in installer' + #13#10#13#10 +
                'After installation, you will need to download models by running:' + #13#10 +
                '  1. Open Command Prompt' + #13#10 +
                '  2. Navigate to installation directory' + #13#10 +
                '  3. Run: python download_models.py' + #13#10#13#10 +
                'This is a one-time download of approximately 600 MB.' + #13#10 +
                'After downloading, Classmark will work completely offline.';
  end;

  ModelsPage.RichEditViewer.Text := InfoText;
end;

function GetUninstallString: String;
var
  sUnInstPath: String;
  sUnInstallString: String;
begin
  sUnInstPath := ExpandConstant('Software\Microsoft\Windows\CurrentVersion\Uninstall\{#emit SetupSetting("AppId")}_is1');
  sUnInstallString := '';
  if not RegQueryStringValue(HKLM, sUnInstPath, 'UninstallString', sUnInstallString) then
    RegQueryStringValue(HKCU, sUnInstPath, 'UninstallString', sUnInstallString);
  Result := sUnInstallString;
end;

function IsUpgrade: Boolean;
begin
  Result := (GetUninstallString() <> '');
end;

function UnInstallOldVersion: Integer;
var
  sUnInstallString: String;
  iResultCode: Integer;
begin
  Result := 0;
  sUnInstallString := GetUninstallString();
  if sUnInstallString <> '' then begin
    sUnInstallString := RemoveQuotes(sUnInstallString);
    if Exec(sUnInstallString, '/SILENT /NORESTART /SUPPRESSMSGBOXES','', SW_HIDE, ewWaitUntilTerminated, iResultCode) then
      Result := 3
    else
      Result := 2;
  end else
    Result := 1;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep=ssInstall) then
  begin
    if (IsUpgrade()) then
    begin
      UnInstallOldVersion();
    end;
  end;
end;

[UninstallDelete]
; Clean up any generated files
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\output"
Type: files; Name: "{app}\*.log"
