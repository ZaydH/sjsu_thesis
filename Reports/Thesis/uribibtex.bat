REM uribibtex.bat version 1.0

@ECHO OFF
IF /I [%1] == [thesis.aux] GOTO runmulti
IF /I [%1] == [thesis] GOTO runmulti
GOTO runnormal

:runmulti
IF EXIST genbib.txt move /Y genbib.txt genbib.bat
IF EXIST genbib.bat (genbib.bat) ELSE (GOTO runnormal)
GOTO end

:runnormal
bibtex build/%1

:end
