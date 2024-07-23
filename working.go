package goregression

import "sync"

type Job func()

type Workers struct {
	sync.WaitGroup
	input chan Job
}

func (w *Workers) Start(size int) {
	w.Wait()
	w.input = make(chan Job)
	for i := 0; i < size; i++ {
		go w.doJobs()
	}
}

func (w *Workers) doJobs() {
	for job := range w.input {
		job()
		w.Done()
	}
}

func (w *Workers) Go(j Job) {
	w.Add(1)
	w.input <- j
}

func (w *Workers) Stop() {
	close(w.input)
}
