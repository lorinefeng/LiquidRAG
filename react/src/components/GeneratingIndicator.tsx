import React from 'react'
import styles from './GeneratingIndicator.module.css'

type Props = {
  visible?: boolean
  text?: string
}

export default function GeneratingIndicator({ visible = true, text = 'Generating' }: Props) {
  const letters = Array.from(text)
  return (
    <div className={`${styles.wrapper} ${visible ? styles.show : ''}`} aria-hidden={!visible}>
      {letters.map((ch, i) => (
        <span key={i} className={styles.letter}>{ch}</span>
      ))}
      <div className={styles.loader} />
    </div>
  )
}