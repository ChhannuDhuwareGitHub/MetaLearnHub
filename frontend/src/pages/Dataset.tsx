import React, { useEffect, useState } from 'react';
import api from '../api';
import { Database, Image as ImageIcon, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function Dataset() {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [selectedBreed, setSelectedBreed] = useState<string>('');
  const [images, setImages] = useState<any[]>([]);
  const [loadingImages, setLoadingImages] = useState(false);

  useEffect(() => {
    api.get('/dataset/stats')
      .then(res => {
        setStats(res.data);
        if (res.data.breeds_found && res.data.breeds_found.length > 0) {
          setSelectedBreed(res.data.breeds_found[0]);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error('Error fetching dataset stats', err);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (selectedBreed) {
      setLoadingImages(true);
      api.get(`/dataset/class/${encodeURIComponent(selectedBreed)}/images`)
        .then(res => {
          setImages(res.data.images || []);
          setLoadingImages(false);
        })
        .catch(err => {
          console.error('Error fetching images', err);
          setLoadingImages(false);
        });
    }
  }, [selectedBreed]);

  if (loading) {
    return <div className="flex justify-center items-center h-64 animate-pulse">Loading Dataset Inspection...</div>;
  }

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-2">Dataset Inspection</h1>
        <p className="text-secondary">Analysis of the raw cattle breed dataset with strict data leakage checks.</p>
      </div>

      <div className="grid grid-cols-4 gap-6 mb-8">
        <div className="stat-card">
          <div className="text-secondary text-sm mb-1">Total Files</div>
          <div className="text-2xl font-bold">{stats?.total_files || 0}</div>
        </div>
        <div className="stat-card">
          <div className="text-secondary text-sm mb-1">Underlying Samples (Groups)</div>
          <div className="text-2xl font-bold text-primary">{stats?.total_sample_groups || 0}</div>
        </div>
        <div className="stat-card">
          <div className="text-secondary text-sm mb-1">Complete Base/_c Pairs</div>
          <div className="text-2xl font-bold text-success">{stats?.total_complete_pairs || 0}</div>
        </div>
        <div className="stat-card">
          <div className="text-secondary text-sm mb-1">Corrupted Files</div>
          <div className="text-2xl font-bold text-danger">{stats?.total_corrupted || 0}</div>
        </div>
      </div>

      <div className="card mb-8 border-l-4 border-l-success">
        <div className="flex items-start gap-4">
          <ShieldCheck className="text-success mt-1" size={24} />
          <div>
            <h3 className="font-semibold text-lg text-success mb-1">Data Leakage Prevention Active</h3>
            <p className="text-sm text-secondary">
              The dataset inspector successfully grouped base images and their "_c" transformed counterparts into {stats?.total_sample_groups} unified sample groups. 
              The Episodic Sampler and Splitter guarantee that a pair will never be split across train/test boundaries.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="col-span-1 card">
          <div className="card-header">
            <h2 className="card-title">Class Distribution</h2>
          </div>
          <div className="flex flex-col gap-2">
            {stats?.breeds_found?.map((breed: string) => {
              const breedStats = stats.breed_stats?.[breed];
              const isActive = selectedBreed === breed;
              return (
                <div 
                  key={breed} 
                  className={`p-3 rounded-md cursor-pointer transition-colors border ${isActive ? 'bg-[var(--primary-light)] border-primary' : 'bg-[var(--bg-main)] border-[var(--border-light)] hover:border-[var(--primary)]'}`}
                  onClick={() => setSelectedBreed(breed)}
                >
                  <div className="font-medium text-sm mb-1">{breed}</div>
                  <div className="flex justify-between text-xs text-secondary">
                    <span>{breedStats?.sample_groups || 0} groups</span>
                    <span>{breedStats?.complete_pairs || 0} pairs</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="col-span-2 card">
          <div className="card-header">
            <h2 className="card-title flex items-center gap-2">
              <ImageIcon size={20} className="text-primary" />
              Sample Images: {selectedBreed}
            </h2>
          </div>
          {loadingImages ? (
            <div className="h-48 flex items-center justify-center">Loading images...</div>
          ) : (
            <div className="image-grid">
              {images.map((img, i) => (
                <div key={i} className="image-card group">
                  <img src={img.data} alt={img.filename} />
                  <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-2">
                    <span className="text-[10px] text-white truncate">{img.filename}</span>
                    <span className="text-[10px] text-blue-300">Group: {img.sample_group_id}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
