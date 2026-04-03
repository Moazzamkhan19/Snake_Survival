# TO COMPILE RUN THIS 

g++ snake.cpp -o snake.exe -IC:\SFML\include -LC:\SFML\lib -lsfml-graphics -lsfml-window -lsfml-system -fopenmp -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi

# TO EXECUTE RUN THIS 

mpiexec -n 4 snake.exe
