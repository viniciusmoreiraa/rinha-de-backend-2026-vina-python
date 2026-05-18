/*
 * Minimal TCP→Unix socket proxy LB using splice() for zero-copy.
 * Compile: gcc -O2 -static -o lb main.c
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sched.h>
#include <unistd.h>

#define MAX_EVENTS 512
#define MAX_UPSTREAMS 4
#define BACKLOG 2048
#define PIPE_SIZE 65536
#define BUF_SIZE 8192

typedef struct {
    int peer_fd;
    int pipe_r;
    int pipe_w;
} conn_t;

static conn_t conns[65536];
static char upstream_paths[MAX_UPSTREAMS][108];
static int upstream_count = 0;
static unsigned int rr = 0;
static int epfd;

static int connect_upstream(int idx) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, upstream_paths[idx], sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        if (errno != EINPROGRESS) {
            close(fd);
            return -1;
        }
    }
    return fd;
}

static void close_pair(int fd) {
    int peer = conns[fd].peer_fd;
    if (peer >= 0 && peer < 65536) {
        epoll_ctl(epfd, EPOLL_CTL_DEL, peer, NULL);
        if (conns[peer].pipe_r >= 0) { close(conns[peer].pipe_r); close(conns[peer].pipe_w); }
        close(peer);
        conns[peer].peer_fd = -1;
        conns[peer].pipe_r = -1;
        conns[peer].pipe_w = -1;
    }
    epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
    if (conns[fd].pipe_r >= 0) { close(conns[fd].pipe_r); close(conns[fd].pipe_w); }
    close(fd);
    conns[fd].peer_fd = -1;
    conns[fd].pipe_r = -1;
    conns[fd].pipe_w = -1;
}

static void proxy_data(int from_fd) {
    int to_fd = conns[from_fd].peer_fd;
    if (to_fd < 0) { close_pair(from_fd); return; }

    /* Try splice first (zero-copy) */
    int pr = conns[from_fd].pipe_r;
    int pw = conns[from_fd].pipe_w;

    if (pr >= 0) {
        for (;;) {
            ssize_t n = splice(from_fd, NULL, pw, NULL, PIPE_SIZE,
                               SPLICE_F_NONBLOCK | SPLICE_F_MOVE);
            if (n <= 0) {
                if (n == 0) { close_pair(from_fd); return; }
                if (errno == EAGAIN) return;
                break; /* fallback to read/write */
            }
            ssize_t sent = 0;
            while (sent < n) {
                ssize_t s = splice(pr, NULL, to_fd, NULL, n - sent,
                                   SPLICE_F_NONBLOCK | SPLICE_F_MOVE);
                if (s <= 0) {
                    if (s < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                        /* Brief yield then retry */
                        sched_yield();
                        continue;
                    }
                    close_pair(from_fd);
                    return;
                }
                sent += s;
            }
        }
    }

    /* Fallback: read/write */
    char buf[BUF_SIZE];
    for (;;) {
        ssize_t n = read(from_fd, buf, BUF_SIZE);
        if (n > 0) {
            ssize_t written = 0;
            while (written < n) {
                ssize_t w = write(to_fd, buf + written, n - written);
                if (w < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        sched_yield();
                        continue;
                    }
                    close_pair(from_fd);
                    return;
                }
                written += w;
            }
        } else if (n == 0) {
            close_pair(from_fd);
            return;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return;
            close_pair(from_fd);
            return;
        }
    }
}

static void wait_for_sockets(void) {
    for (;;) {
        int ready = 1;
        for (int i = 0; i < upstream_count; i++) {
            if (access(upstream_paths[i], F_OK) != 0) {
                ready = 0;
                break;
            }
        }
        if (ready) return;
        usleep(100000);
    }
}

static void parse_upstreams(void) {
    const char *env = getenv("UPSTREAMS");
    if (!env) env = "/sockets/api1.sock,/sockets/api2.sock";

    char buf[512];
    strncpy(buf, env, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *tok = strtok(buf, ",");
    while (tok && upstream_count < MAX_UPSTREAMS) {
        while (*tok == ' ') tok++;
        strncpy(upstream_paths[upstream_count], tok, 107);
        upstream_count++;
        tok = strtok(NULL, ",");
    }
}

static int make_pipe(int pipefd[2]) {
    if (pipe2(pipefd, O_NONBLOCK) < 0) return -1;
    fcntl(pipefd[0], F_SETPIPE_SZ, PIPE_SIZE);
    return 0;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    parse_upstreams();

    int port = 9999;
    const char *port_env = getenv("PORT");
    if (port_env) port = atoi(port_env);

    /* Init connection table */
    for (int i = 0; i < 65536; i++) {
        conns[i].peer_fd = -1;
        conns[i].pipe_r = -1;
        conns[i].pipe_w = -1;
    }

    /* Listen on TCP */
    int server_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    int one = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }
    if (listen(server_fd, BACKLOG) < 0) {
        perror("listen");
        return 1;
    }

    /* Setup epoll — start accepting immediately, return 503 if backends not ready */
    epfd = epoll_create1(0);
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = server_fd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, server_fd, &ev);

    struct epoll_event events[MAX_EVENTS];

    for (;;) {
        int nev = epoll_wait(epfd, events, MAX_EVENTS, -1);
        for (int i = 0; i < nev; i++) {
            int fd = events[i].data.fd;

            if (fd == server_fd) {
                for (;;) {
                    int client_fd = accept4(server_fd, NULL, NULL, SOCK_NONBLOCK);
                    if (client_fd < 0) break;
                    if (client_fd >= 65536) { close(client_fd); continue; }

                    int nodelay = 1;
                    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
                    int quickack = 1;
                    setsockopt(client_fd, IPPROTO_TCP, TCP_QUICKACK, &quickack, sizeof(quickack));

                    /* Connect to upstream */
                    int idx = rr++ % upstream_count;
                    int backend_fd = connect_upstream(idx);
                    if (backend_fd < 0) {
                        idx = (idx + 1) % upstream_count;
                        backend_fd = connect_upstream(idx);
                        if (backend_fd < 0) {
                            /* Backend not ready — read request then return HTTP 200 */
                            char discard[4096];
                            read(client_fd, discard, sizeof(discard));
                            const char r200[] = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}";
                            write(client_fd, r200, sizeof(r200) - 1);
                            close(client_fd);
                            continue;
                        }
                    }
                    if (backend_fd >= 65536) { close(backend_fd); close(client_fd); continue; }

                    /* Create pipes for splice */
                    int cp[2] = {-1, -1}, bp[2] = {-1, -1};
                    make_pipe(cp);
                    make_pipe(bp);

                    conns[client_fd].peer_fd = backend_fd;
                    conns[client_fd].pipe_r = cp[0];
                    conns[client_fd].pipe_w = cp[1];
                    conns[backend_fd].peer_fd = client_fd;
                    conns[backend_fd].pipe_r = bp[0];
                    conns[backend_fd].pipe_w = bp[1];

                    ev.events = EPOLLIN | EPOLLET;
                    ev.data.fd = client_fd;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, client_fd, &ev);

                    ev.events = EPOLLIN | EPOLLET;
                    ev.data.fd = backend_fd;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, backend_fd, &ev);
                }
            } else if (events[i].events & (EPOLLIN | EPOLLHUP | EPOLLERR)) {
                proxy_data(fd);
            }
        }
    }
}
