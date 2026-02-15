# PyInstaller spec. Сборка: pyinstaller build_exe.spec
# Результат: dist/PlantClassification/ — папка с PlantClassification.exe и DLL.
# В релиз загрузить zip этой папки или саму папку.

block_cipher = None

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'torch', 'torchvision', 'timm', 'PIL', 'numpy', 'pandas',
        'sklearn.metrics', 'sklearn.metrics._classification',
        'tqdm', 'matplotlib', 'seaborn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PlantClassification',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PlantClassification',
)
