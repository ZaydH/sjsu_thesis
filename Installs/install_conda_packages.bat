@echo off 
echo.
echo.
echo.
echo.Updating Python to the latest 2.x version
conda update python
echo.Python update complete.

echo. 
echo.  
echo.Installing Python packages.
call:conda_install pip -y
call:conda_install Pillow -y
call:conda_install enum34 -y
call:conda_install docutils -y
call:conda_install sphinx -y
call:conda_install numpy -y
call:conda_install scikit-image -y
call:conda_install scipy -y
call:conda_install scikit-learn -y

:conda_install_from_conda_direct opencv3
:conda_install_from_conda_direct opencv2


echo.
echo.
echo.End of Package Installation.
pause
goto:eof


:conda_install
echo.
echo.
echo.
echo.Installing package: %~1
conda install %~1 -y
echo.
goto:eof


:conda_install_from_conda_direct
echo.
echo.
echo.
echo.Installing package: %~1
conda install -c https://conda.anaconda.org/menpo %~1
echo.
goto:eof