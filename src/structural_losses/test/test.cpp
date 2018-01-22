#include <vector>
#include <iostream>
#include <stdio.h>
#include <math.h>

void print_vec(int n, int m, const std::vector<double>& w)
{
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<m;j++)
            std::cout << w[i*m + j] << " ";
        printf("\n");
    }
}
void print_mat(int n, int m, float* mat)
{
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<m;j++)
            printf("%3.1f ", mat[i*m+j]);
        printf("\n");
    }
}

void approxmatch_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,float * match){
	for (int i=0;i<b;i++){
        // 
        int factorl=std::max(n,m)/n; //1
        int factorr=std::max(n,m)/m; //1
        // 饱和度
        std::vector<double> saturatedl(n,double(factorl));
        std::vector<double> saturatedr(m,double(factorr));

		std::vector<double> weight(n*m);
        // match [nxm] = {0}
        for (int j=0;j<n*m;j++)
			match[j]=0;
        //
		for (int j=8;j>=-2;j--) // 
        {
			//printf("i=%d j=%d\n",i,j);
			// level 只和 循环次数 有关系
            double level=-powf(4.0,j);
			if (j==-2)
				level=0;
            printf("------level %lf-------\n", level);
            // 求了个权重 等于 level x 欧式距离 x 饱和度right
            for (int k=0;k<n;k++)
            {
				double x1=xyz1[k*3+0];
				double y1=xyz1[k*3+1];
				double z1=xyz1[k*3+2];
				for (int l=0;l<m;l++)
                {
					double x2=xyz2[l*3+0];
					double y2=xyz2[l*3+1];
					double z2=xyz2[l*3+2];
                    // wt-> 初始化权重
					weight[k*m+l]=expf(level*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))) * saturatedr[l];
				}
			}
            // 按照行 归一化了权重 又乘以 饱和度left
			std::vector<double> ss(m,1e-9);
			for (int k=0;k<n;k++){
				double s=1e-9;
				for (int l=0;l<m;l++){
					s+=weight[k*m+l];
				}
				for (int l=0;l<m;l++){
                    // wt-> 按照行 归一化权重乘以 饱和度
					weight[k*m+l] = weight[k*m+l]/s *saturatedl[k];
				}
                //权重 按照列求的和 
				for (int l=0;l<m;l++)
					ss[l]+=weight[k*m+l];
			}
            printf("------weight 1-------\n");
            print_vec(n,m,weight);
            // ss 是权重 按照列求的和  进行了一些变化
			for (int l=0;l<m;l++){
				double s=ss[l];
				double r=std::min(saturatedr[l]/s,1.0);
				ss[l]=r;
			}
            // ss乘到这一列的每个元素上
            // s 按照行 求和
            // ss2 按照列 求和
            // 更新饱和度left
			std::vector<double> ss2(m,0);
			for (int k=0;k<n;k++)
            {
				double s=0;
				for (int l=0;l<m;l++){
                    // wt-> 权重 乘以 ss2 每一列的和
					weight[k*m+l]*=ss[l];
					s+=weight[k*m+l];
					ss2[l]+=weight[k*m+l];
				}
                // st-> 饱和度left 减去 权重按照行求和
				saturatedl[k]=std::max(saturatedl[k]-s,0.0);
			}
            printf("------weight 2-------\n");
            print_vec(n,m,weight);
            // 匹配的权重 累加的记录到 match里面
			for (int k=0;k<n*m;k++)
				match[k]+=weight[k];
            printf("------match-------\n");
            print_mat(n,m,match);
            // 更新饱和度right
			for (int l=0;l<m;l++){
                // st-> 饱和度right 减去 权重按照列求和
				saturatedr[l]=std::max(saturatedr[l]-ss2[l],0.0);
			}
            printf("----S_left, S_right 1----\n");
            print_vec(1, n, saturatedl);
            print_vec(1, m, saturatedr);
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
	}
}

int main()
{
    float mt[4*4];
    float pc1[4*3] = { 0,0,0, 10,0,0,  0,10,0, 0,0,10};
    float pc2[4*3] = {1,11,0,  1,1,1,  0,0,11, 10,0,0};
    approxmatch_cpu(1, 4, 4, pc1, pc2, mt);
    
    print_mat(4,4,mt);
    
    return 0;
}
