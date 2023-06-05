#include<iostream>
#include "mpi.h"
#include <sys/time.h>
#include <omp.h>
#include<arm_neon.h>
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
int main(){

	int N;N=n;
	struct timeval beg1,end1;
	Initialize(N);
	int rank,size,r1,r2;
	int i,j,k;
    float32_t tmp[4];
    float32x4_t t1,t2,t3,t4;
    MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

            gettimeofday(&beg1, NULL);

                if(n%size==0){
                    r1=rank*(n/size);
                    r2=(rank==size-1)?n-1:(r1+n/size-1);
                }
                else{
                    r1=rank*(n/size+1);
                    r2=(rank==size-1)?n-1:(r1+n/size);
                }
                if(rank==0){
                    for (j=1;j<size;j++){
                        int t1,t2;
                        if(n%size==0){
                            t1=j*(n/size);
                            t2=(j==size-1)?n-1:(t1+n/size-1);
                        }
                        else{
                            t1=j*(n/size+1);
                            t2=(j==size-1)?n-1:(t1+n/size);
                        }
                        MPI_Send(&B[t1][0],n*(t2-t1+1),MPI_FLOAT,j,0,MPI_COMM_WORLD);
                    }
                }
                else
                    MPI_Recv(&B[r1][0],n*(r2-r1+1),MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    MPI_Barrier(MPI_COMM_WORLD);
                #pragma omp parallel num_threads(threads),private(i,j,k,tmp,t1,t2,t3,t4)
                for(k=0;k<n;k++){
                    if(rank==0)
                    {
                        #pragma omp single
                        {
                            tmp[0]=B[k][k];tmp[1]=B[k][k];tmp[2]=B[k][k];tmp[3]=B[k][k];
                            t1=_mm_loadu_ps(tmp);
                            for(j=k+1;j+4<=n;j+=4){
                                t2=vld1q_f32(B[k]+j); //把内存中从B[k][j]开始的四个单精度浮点数加载到t2寄存器
                                t3=vdivq_f32(t2,t1); //相除结果放到t3寄存器
                                vst1q_f32(B[k]+j,t3); //把t3寄存器的值放回内存
                            }

                            for(j;j<n;j++) //处理剩下的不能被4整除的
                                B[k][j]/=B[k][k];
                            B[k][k]=1.0;
                        }
                        for(j=1;j<size;j++)
                            MPI_Send(&B[k][0],n,MPI_FLOAT,j,k+1,MPI_COMM_WORLD);

                    }
                    else
                        MPI_Recv(&B[k][0],n,MPI_FLOAT,0,k+1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    if(r2>=k+1){
                        #pragma omp for
                        for(i=max(k+1,r1);i<=r2;i++){
                            tmp[0]=B[i][k];tmp[1]=B[i][k];tmp[2]=B[i][k];tmp[3]=B[i][k];
                            t1=vld1q_f32(tmp);
                            for(j=k+1;j+4<=n;j+=4){
                                t2=vld1q_f32(B[k]+j);
                                t3=vld1q_f32(B[i]+j);
                                t4=vmulq_f32(t1,t2);
                                t3=vsubq_f32(t3,t4);
                                vst1q_f32(B[i]+j,t3);
                            }
                            for(j=j;j<n;j++)
                                B[i][j]-=B[i][k]*B[k][j];
                            B[i][k]=0;
                                    }
                    }
                    if(rank!=0)
                        MPI_Send(&B[r1][0],n*(r2-r1+1),MPI_FLOAT,0,2,MPI_COMM_WORLD);
                    else
                        for(int q=1;q<size;q++){
                            int beg=max(k+1,r1);
                            MPI_Recv(&B[beg][0],n*(r2-beg+1),MPI_FLOAT,q,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        }
            }

            gettimeofday(&end1, NULL);
            cout<<n<<" "<<(long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec<<endl;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}