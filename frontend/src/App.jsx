import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Analyzer from './pages/Analyzer';
import HowItWorks from './pages/HowItWorks';
import About from './pages/About';
import './styles/global.css';

export default function App() {
  const [currentPage, setCurrentPage] = useState('home');

  return (
    <>
      <Navbar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      
      {currentPage === 'home' && <Home setCurrentPage={setCurrentPage} />}
      {currentPage === 'analyzer' && <Analyzer />}
      {currentPage === 'how-it-works' && <HowItWorks setCurrentPage={setCurrentPage} />}
      {currentPage === 'about' && <About />}

      <Footer setCurrentPage={setCurrentPage} />
    </>
  );
}
