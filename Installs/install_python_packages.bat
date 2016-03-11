SET pip_install = "python -m pip install"
@echo off 
call:pip_install pip 
call:pip_install Pillow
call:pip_install enum34
call:pip_install docutils
call:pip_install sphinx
call:pip_install numpy


:: End of installation.
echo.
echo.
echo. End of Package Installation.
pause
goto:eof

::----------------------------------------------------------------------
::   Moving pip install to a function so the call is standardized
::----------------------------------------------------------------------

:pip_install
echo. Installing package: %~1
python -m pip install %~1 --upgrade
echo.
goto:eof