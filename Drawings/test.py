import win32com.client
win32com.client.gencache.EnsureModule(
    '{83A33D31-27C5-11CE-BFD4-00400513BB57}',  # SolidWorks type library GUID
    0,  # LCID (default locale)
    1,  # Major version (1 works for most)
    0   # Minor version
)