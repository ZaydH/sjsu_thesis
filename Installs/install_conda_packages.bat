@echo off 
echo.
echo.
echo.
echo.Updating Python to the latest 2.x version
conda update python -y
echo.Python update complete.

echo. 
echo.  
echo.Installing Python packages.
call:conda_basic_install pip
call:conda_basic_install Pillow
call:conda_basic_install enum34
call:conda_basic_install docutils
call:conda_basic_install sphinx
call:conda_basic_install numpy
call:conda_basic_install scikit-image
call:conda_basic_install scipy
call:conda_basic_install scikit-learn
call:conda_basic_install pylint

call:conda_install_from_conda_direct opencv3
call:conda_install_from_conda_direct opencv


echo.
echo.
echo.End of Package Installation.
pause
goto:eof


:conda_basic_install
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
conda install -c https://conda.anaconda.org/menpo %~1 -y
echo.
goto:eof