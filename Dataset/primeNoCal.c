#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<error.h>
#include<sys/time.h>
int main(int argc , char **argv){
	int process_no, i, limit, shift, start, end, counter, number, status=0;
	pid_t child_id,wpid;
	struct timeval starttime , endtime;
	double start_ms , end_ms , diff;
	gettimeofday(&starttime , NULL);
	if(argc == 1){
		printf("Enter a command line argument.\n");
		return 0;
	}else{
		number = atoi(argv[1]);
		printf("Enter number of process :");
		scanf("%d", &process_no);
		limit = sqrt(number)+1;
		shift = (limit-2) / process_no;
		//Loop forking n processes.
		for(i=0; i<process_no; i++){
				if(i == 0){
					start = 2;
				}else{
					start = end + 1;
				}
				end = start + shift;
				if(start >= limit-1) break;
				child_id = fork();
				if(child_id == 0){
				 	for(counter = start; counter <= end; counter++){
						if(number % counter == 0){
							number = number/counter;
							printf("Prime factors : %d", counter);
							printf(" %d\n", number);
							exit(0);
						}
					}
				 	exit(0);
				}
		}
		// parent waiting the childs quit execution.
		while ((wpid = wait(&status)) > 0);
		
		gettimeofday(&endtime , NULL);
		start_ms = (double)starttime.tv_sec*1000000 + (double)starttime.tv_usec;
	    end_ms = (double)endtime.tv_sec*1000000 + (double)endtime.tv_usec;
	    diff = (double)end_ms - (double)start_ms;
		printf("Time : %lf microseconds.\n", diff);
		
	}
	return 0;
}
