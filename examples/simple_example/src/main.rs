use deepthought::DeepThoughtBuilder;

pub const DOCUMENT: &str = r#"
Leo Tolstoy was born at Yasnaya Polyana, in Russia's Tula Province, the fourth of five children. The title of Count had been conferred on his ancestor in the early 18th century by Peter the Great. His parents died when he was a child, and he was brought up by relatives. In 1844 Tolstoy started his studies of law and oriental languages at Kazan University, but he never took a degree. Dissatisfied with the standard of education, he returned back to Yasnaya Polyana in the middle of his studies, and then spent much of his time in Moscow and St. Petersburg. In 1847 Tolstoy was treated for venereal disease. After contracting heavy gambling debts, Tolstoy accompanied his elder brother to the Caucasus in 1851, and joined an artillery regiment. In the 1850s Tolstoy also began his literary career, publishing the autobiographical trilogy Childhood (1852), Boyhood (1854), and Youth (1857).

During the Crimean War Tolstoy commanded a battery, witnessing the siege of Sebastopol (1854-55). In 1857 he visited France, Switzerland, and Germany. After his travels Tolstoy settled in Yasnaya Polyana, where he started a school for peasant children. He saw that the secret of changing the world lay in education. He investigated educational theory and practice during further travels to Europe (1860-61), and published magazines and textbooks on the subject. In 1862 he married Sonya Andreyevna Behrs (1844-1919); she bore him 13 children. Sonya also acted as her husband's devoted secretary.

Tolstoy's fiction grew originally out of his diaries, in which he tried to understand his own feelings and actions so as to control them. He read fiction and philosophy widely. In the Caucasus he read Plato and Rousseau, Dickens and Sterne; through the 1850s he also read and admired Goethe, Stendhal, Thackeray, and George Eliot.

Tolstoy's major work, War and Peace, appeared between the years 1865 and 1869. The epic tale depicted the story of five families against the background of Napoleon's invasion of Russia. Its vast canvas includes 580 characters, many historical, others fictional. War and Peace reflected Tolstoy's view that all is predestined, but we cannot live unless we imagine that we have free will.

Tolstoy's other masterpiece, Anna Karenina (1873-77), told a tragic story of a married woman, who follows her lover, but finally at a station throws herself in front of an incoming train. The novel opens with the famous sentence: "Happy families are all alike, every unhappy family is unhappy in its own way." Anna Karenina has been filmed in Hollywood several times. One of the most famous versions, starring Greta Garbo, was born during the period when film industry was under the censorial agencies of the Catholic Legion of Decency and the Production Code Administration. Thus the love affair of Anna and Vronsky was strongly condemned in the film and all references to the illegitimate child were removed.

After finishing Anna Karenina Tolstoy renounced all his earlier works. "I wrote everything into Anna Karenina," he later confessed, "and nothing was left over." Voskresenie (1899, Resurrection) was Tolstoy's last major novel, which affirmed Tolstoy's belief in the primacy of the individual conscience over the collective morality of the group.

According to Tolstoy's wife Sonia, the idea for The Kreutzer Sonata (1890) was given to Tolstoy by the actor V.N. Andreev-Burlak during his visit at Yasnaya Polyana in June 1887. In the spring of 1888 an amateur performance of Beethoven's Kreutzer Sonata took place in Tolstoy's home and it made the author to return to an idea he had had in the 1860s. The Kreutzer Sonata is written in the form of a frame-story and set on a train. The conversations among the passengers develop into a discussion of the institution of marriage. After writing the novel Tolstoy was accused of preaching immorality. The Chief Procurator of the Holy Synod wrote to the tsar, and this marked the beginning of the process that led ultimately to Tolstoy's excommunication. In 1890, Tolstoy was forced to write a postscript in which he attempted to explain his unorthodox views.

In the 1880s Tolstoy wrote such philosophical works as A Confession and What I Believe, which was banned in 1884. He started to see himself more as a sage and moral leader than an artist. In 1884 occurred his first attempt to leave home. He gave up his estate to his family, and tried to live as a poor, celibate peasant. Attracted by Tolstoy's writings, Yasnaya Polyana was visited by hundreds of people from all over the world. In 1901 the Russian Orthodox Church excommunicated the author. Tolstoy became seriously ill and he recuperated in Crimea.

Tolstoy's teachings influenced Gandhi in India, and the kibbutz movement in Palestine, and in Russia his moral authority rivaled that of the tsar. After leaving his estate with his disciple Vladimir Chertkov on the urge to live as a wandering ascetic, Tolstoy died of pneumonia on November 7 (Nov. 20, New Style) in 1910, at a remote railway junction. Tolstoy's collected works, which were published in the Soviet Union in 1928-58, consisted of 90 volumes.
"#;

fn main() {
    let mut dt = DeepThoughtBuilder::new()
        .chat_model_gguf("../../Llama-3.2-3B-Instruct-Q6_K.gguf".to_string())
        .embed_model_gguf("../../nomic-embed-text-v1.Q5_K_M.gguf".to_string())
        .chunk_size(1024)
        .chunk_overlap(16)
        .embedding_doc_prefix("search_document".to_string())
        .embedding_query_prefix("search_query".to_string())
        .build()
        .unwrap();
    println!(
        "Embedding size = {}",
        dt.embed("Hello world").unwrap()[0].len()
    );
    dt.add_document(DOCUMENT).unwrap();
    dt.sync().unwrap();
    let res = dt.query("Who is Anna Karenina?").unwrap();
    for doc in res.iter() {
        println!("{}", &doc);
        println!(">>>>>>>>>>>>>>>>");
    }
}
