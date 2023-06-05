#include<iostream>
#include "mpi.h"
#include <sys/time.h>
#include <omp.h>
#include <stdlib.h>
using namespace std;
float A[5000][5000],B[5000][5000];
int n=2048;
int threads=4;
void Initialize(int n)//初始化
{
	int i,j,k;
	for(i=0;i<n;i++)
	{
		for(j=0;j<i;j++){
			A[i][j]=0;//下三角元素初始化为零
			B[i][j]=0;

		}
		A[i][i]=1.0;//对角线元素初始化为1
		B[i][i]=1.0;

		for(j=i+1;j<n;j++){
			A[i][j]=rand();//上三角元素初始化为随机数
			B[i][j]=A[i][j];

		}
	}
	for(k=0;k<n;k++)
		for(i=k+1;i<n;i++)
			for(j=0;j<n;j++){
				A[i][j]+=A[k][j];//最终每一行的值是上一行的值与这一行的值之和
				B[i][j]+=B[k][j];

			}
}
void Print(int n,float m[][2000]){//打印结果
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++)
			cout<<m[i][j]<<" ";
		cout<<endl;
	}
}
void Gauss_normal(){//串行算法
	int i,j,k;
	for(k=0;k<n;k++)
    {
		float tmp=A[k][k];
		for(j=k+1;j<n;j++)
			A[k][j]/=tmp;
        A[k][k]=1;
		for(i=k+1;i<n;i++)
		{
			float tmp2=A[i][k];
			for(j=k+1;j<n;j++)
				A[i][j]-=tmp2*A[k][j];
			A[i][k]=0;
		}
	}
}
int main(){

	int N;N=n;
	struct timeval beg1,end1;
	Initialize(N);
	int i,j,k,size,rank;
    float tmp;
    MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

            gettimeofday(&beg1, NULL);

                if(rank==0){
                    for (j=1;j<size;j++){
                        for(i=j;i<n;i+=size)
                            MPI_Send(&B[i][0],n,MPI_FLOAT,j,i,MPI_COMM_WORLD);
                    }
                }
                else
                    for(i=rank;i<n;i+=size)
                        MPI_Recv(&B[i][0],n,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                for(k=0;k<n;k++){
                    if(rank==0)
                    {
                            tmp=B[k][k];
                            for(j=k+1;j<n;j++)
                                B[k][j]/=tmp;
                            B[k][k]=1;
                        for(j=1;j<size;j++)
                            MPI_Send(&B[k][0],n,MPI_FLOAT,j,k+1,MPI_COMM_WORLD);

                    }
                    else
                        MPI_Recv(&B[k][0],n,MPI_FLOAT,0,k+1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    if(rank!=0){
                        int r=rank;
                        while(r<k+1)
                            r+=size;
                        for(i=r;i<=n;i+=size){
                            tmp=B[i][k];
                            for(j=k+1;j<n;j++)
                                B[i][j]-=tmp*B[k][j];
                            B[i][k]=0;
                            if (i==k+1&&rank!= 0)
                                MPI_Send(&B[i][0],n,MPI_FLOAT,0,0,MPI_COMM_WORLD);
                        }
                    }
                    else if(k<n-1)
                        MPI_Recv(&B[k+1][0],n,MPI_FLOAT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            gettimeofday(&end1, NULL);
            cout<<n<<" "<<(long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec<<endl;
    MPI_Finalize();
    return 0;
}