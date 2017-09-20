/*
	File: timing.h
    Author(s):
		Cody Balos
	Description:
    	Useful stuff for timing programs.

    MIT License

    Copyright (c) [2017] [Cody Balos]

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef TIMING_H
#define TIMING_H

#include <sys/time.h>

inline long get_elapsed_time(struct timeval start, struct timeval end);

#define TIME_BLOCK_EXEC(msg, ...) do {                                           \
    struct timeval __start, __end;                                               \
    gettimeofday(&__start, NULL);                                                \
    __VA_ARGS__                                                                  \
    gettimeofday(&__end, NULL);                                                  \
    printf("%s\n%ldms\n", msg, get_elapsed_time(__start, __end));                \
} while(0);

/// Returns the elapsed time between two timeval in milliseconds.
long get_elapsed_time(struct timeval start, struct timeval end)
{
	return ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))/1000.0;
}

#endif