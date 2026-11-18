/* TODO: Add your ID and full name before submission. */
#include "pagerank.h"
#include "thr_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    Graph *graph;
    size_t count;
    size_t worker_count;
    size_t *outdegree;
    float *rank;
    double *contributions;
    double dangling;
} PageRankState;

typedef struct {
    PageRankState *state;
    /* This worker handles vertices in [begin, end). */
    size_t begin;
    size_t end;
    size_t *seen;
    double *contributions;
    double dangling;
} PageRankWorker;

static void *allocate_array(size_t count, size_t size)
{
    void *memory = calloc(count, size);
    if (memory == NULL) {
        perror("PageRank allocation");
        abort();
    }
    return memory;
}

static void *count_outgoing(void *argument)
{
    PageRankWorker *worker = argument;
    PageRankState *state = worker->state;

    for (size_t source = worker->begin; source < worker->end; ++source) {
        /* Mark destinations with source + 1 so duplicates count only once.
         * Zero means unseen, so the same array can serve every source. */
        size_t stamp = source + 1;
        for (node *edge = state->graph->adjacencyLists[source];
             edge != NULL; edge = edge->next) {
            size_t destination = (size_t)edge->v;
            if (destination != source && worker->seen[destination] != stamp) {
                worker->seen[destination] = stamp;
                ++state->outdegree[source];
            }
        }
        state->rank[source] = 1.0f / state->count;
    }
    return NULL;
}

static void *distribute_ranks(void *argument)
{
    PageRankWorker *worker = argument;
    PageRankState *state = worker->state;

    /* Source stamps repeat each round, so discard the previous marks. */
    memset(worker->seen, 0, state->count * sizeof(*worker->seen));
    memset(worker->contributions, 0,
           state->count * sizeof(*worker->contributions));
    worker->dangling = 0.0;

    for (size_t source = worker->begin; source < worker->end; ++source) {
        if (state->outdegree[source] == 0) {
            /* This rank will be shared equally among all vertices. */
            worker->dangling += state->rank[source];
            continue;
        }
        double contribution = (double)state->rank[source] / state->outdegree[source];
        size_t stamp = source + 1;
        for (node *edge = state->graph->adjacencyLists[source];
             edge != NULL; edge = edge->next) {
            size_t destination = (size_t)edge->v;
            if (destination != source && worker->seen[destination] != stamp) {
                worker->seen[destination] = stamp;
                worker->contributions[destination] += contribution;
            }
        }
    }
    return NULL;
}

static void *update_ranks(void *argument)
{
    PageRankWorker *worker = argument;
    PageRankState *state = worker->state;
    /* Every vertex gets the teleport probability and a share of dangling rank. */
    double base = (0.15 + 0.85 * state->dangling) / state->count;

    for (size_t destination = worker->begin; destination < worker->end; ++destination) {
        double incoming = 0.0;
        /* Combine the contributions collected by each worker. */
        for (size_t index = 0; index < state->worker_count; ++index) {
            incoming += state->contributions[index * state->count + destination];
        }
        state->rank[destination] = (float)(base + 0.85 * incoming);
    }
    return NULL;
}

static void run_jobs(thr_pool_t *pool, PageRankWorker *workers, size_t count,
                     void *(*job)(void *))
{
    for (size_t index = 0; index < count; ++index) {
        if (thr_pool_queue(pool, job, &workers[index]) != 0) {
            perror("PageRank thread pool queue");
            abort();
        }
    }
    /* The caller can reuse job arguments and results after this returns. */
    thr_pool_wait(pool);
}

void PageRank(Graph *g, int iterations, float *rank)
{
    if (g == NULL || g->numVertices == 0 || rank == NULL) {
        return;
    }

    size_t count = g->numVertices;
    long processors = sysconf(_SC_NPROCESSORS_ONLN);
    size_t worker_count = processors > 1 ? (size_t)processors : 2;
    if (worker_count > count) {
        worker_count = count;
    }

    PageRankState state = {0};
    state.graph = g;
    state.count = count;
    state.worker_count = worker_count;
    state.rank = rank;
    state.outdegree = allocate_array(count, sizeof(*state.outdegree));
    /* Each worker writes its own row, avoiding locks on destination vertices. */
    state.contributions = allocate_array(count, worker_count * sizeof(double));
    PageRankWorker *workers = allocate_array(worker_count, sizeof(*workers));
    thr_pool_t *pool = thr_pool_create(0, (uint32_t)worker_count, 1, NULL);
    if (pool == NULL) {
        perror("PageRank thread pool creation");
        abort();
    }

    size_t begin = 0;
    for (size_t index = 0; index < worker_count; ++index) {
        workers[index].state = &state;
        workers[index].begin = begin;
        /* Give one extra vertex to each of the first count % worker_count jobs. */
        begin += count / worker_count + (index < count % worker_count);
        workers[index].end = begin;
        workers[index].seen = allocate_array(count, sizeof(size_t));
        workers[index].contributions = state.contributions + index * count;
    }

    run_jobs(pool, workers, worker_count, count_outgoing);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        /* All reads finish before any rank is overwritten in the update phase. */
        run_jobs(pool, workers, worker_count, distribute_ranks);
        state.dangling = 0.0;
        for (size_t index = 0; index < worker_count; ++index) {
            state.dangling += workers[index].dangling;
        }
        /* Workers now write disjoint ranges of rank. run_jobs waits before
         * the next iteration starts reading those values. */
        run_jobs(pool, workers, worker_count, update_ranks);
    }

    thr_pool_destroy(pool);
    for (size_t index = 0; index < worker_count; ++index) {
        free(workers[index].seen);
    }
    free(workers);
    free(state.contributions);
    free(state.outdegree);
}
